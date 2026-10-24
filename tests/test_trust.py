"""Trust features: hash-chained audit log, EXIF editing flag, reply checks, abuse / legal escalation,
lesser-of rule and price provenance."""
import io
import json

import negotiation as neg
import pricing_rules as pr
from test_autosettle import (  # noqa: F401
    SCRATCH_CLEAR, SWIFT, _defaults, api, approved_quote, demo_api, demo_token, evidence_stub, item, jpeg, proof,
    scene, upload)


def edited_jpeg(seed=0, software="Adobe Photoshop 25.0 (Windows)"):
    from PIL import Image
    ex = Image.Exif()
    ex[305] = software
    buf = io.BytesIO()
    scene(seed).save(buf, "JPEG", quality=90, exif=ex)
    return buf.getvalue()


class TestAuditChain:
    def test_chain_verifies_and_detects_edits(self, api):
        client, h, srv = api
        upload(client, h, img=jpeg(300))
        upload(client, h, img=jpeg(301))
        v = srv.verify_audit_chain()
        assert v["ok"] and v["checked"] >= 4
        path = srv.CLAIMS_DIR / "audit.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        rec = json.loads(lines[1])
        rec["status"] = "APPROVED"                             # someone edits a past decision
        path.write_text("\n".join(lines[:1] + [json.dumps(rec)] + lines[2:]) + "\n", encoding="utf-8")
        bad = srv.verify_audit_chain()
        assert not bad["ok"] and bad["first_bad_line"] == 2 and "edited" in bad["problem"]
        path.write_text("\n".join(lines[:1] + lines[2:]) + "\n", encoding="utf-8")   # someone deletes a line
        assert "broken" in srv.verify_audit_chain()["problem"]

    def test_staff_sees_chain_status(self, api):
        client, h, srv = api
        upload(client, h, img=jpeg(302))
        tok = client.post("/api/admin/login", json={"username": "admin", "password": "test-pass"}).json()["token"]
        o = client.get("/api/admin/overview", headers={"Authorization": f"Bearer {tok}"}).json()
        assert o["audit_chain"]["ok"] and len(o["audit_chain"]["head"]) == 64


class TestEditedPhotos:
    def test_edited_claim_photo_goes_to_a_person(self, api):
        client, h, srv = api
        j = client.post("/api/analyze_claim", headers=h, files={"file": ("a.jpg", edited_jpeg(303), "image/jpeg")}).json()
        assert j["quote"]["status"] == "ESCALATED" and j["offer"]["current_offer_inr"] is None
        assert any("image-editing" in r for r in j["quote"]["plain_reasons"])
        meta = srv.get_claim(j["claim_id"])["meta"]
        assert meta["photo_integrity"]["software"].startswith("Adobe Photoshop")

    def test_edited_demo_photo_gets_no_partial_offer(self, demo_api):
        client, _ = demo_api
        j = upload(client, demo_token(client), img=edited_jpeg(304, "Snapseed 2.0")).json()
        assert j["offer"]["current_offer_inr"] is None

    def test_camera_photo_and_plain_photo_are_fine(self, api):
        client, h, srv = api
        j = client.post("/api/analyze_claim", headers=h,
                        files={"file": ("a.jpg", edited_jpeg(305, "Android 14 / Pixel 8"), "image/jpeg")}).json()
        assert not any("image-editing" in r for r in j["quote"]["plain_reasons"])

    def test_edited_proof_photo_is_refused(self, demo_api, monkeypatch):
        client, _ = demo_api
        evidence_stub(monkeypatch, SCRATCH_CLEAR)
        h = demo_token(client)
        c0 = upload(client, h, img=jpeg(306)).json()
        c1 = proof(client, h, c0["claim_id"], edited_jpeg(307, "Canva")).json()
        assert c1["offer"]["current_offer_inr"] == c0["offer"]["current_offer_inr"]
        assert "image-editing app" in c1["chat"]["messages"][-1]["text"]

    def test_edited_flag_survives_repricing(self):
        q = pr.price_claim([item("car door")], SWIFT, suspicious=["photo metadata says it was saved by an image editor (GIMP)"])
        assert q["status"] == "ESCALATED" and q["blockers"][0]["code"] == "PHOTO_EDITED"


class TestChatGuards:
    def test_abuse_and_legal_threats_go_to_a_person(self):
        for m in ("you idiots, this is a scam", "main consumer court jaunga", "I will file an FIR",
                  "bewakoof bana rahe ho", "my vakil will send a legal notice"):
            q = approved_quote()
            s = neg.new_session("c", q)
            r = neg.handle_message(s, q, m)
            assert r["outcome"] == "ESCALATED" and r["offer_inr"] == s["offer_inr"], m

    def test_reply_checks(self):
        q = approved_quote()
        tmpl = "Your offer is Rs 2,400."
        allowed = {2400}
        ok = neg.verify_reply("Happy to help - your offer is Rs 2,400.", tmpl, allowed, q)
        assert all(c["passed"] for c in ok)
        for bad, name in [("Your offer is Rs 2,400 and we can go up to the maximum.", "no_internal_limits"),
                          ("Your offer is Rs 2,400, paid within 2 days, guaranteed.", "no_promises"),
                          ("Your offer is Rs 2,400 for the bonnet too.", "only_claim_parts"),
                          ("Your offer is Rs 2,400; this looks like fraud.", "no_accusation"),
                          ("Your offer is Rs 2,900.", "amounts_unchanged")]:
            res = {c["check"]: c["passed"] for c in neg.verify_reply(bad, tmpl, allowed, q)}
            assert res[name] is False, (bad, name)

    def test_rejected_llm_text_is_recorded_for_staff(self, monkeypatch):
        q = approved_quote()
        s = neg.new_session("c", q)
        monkeypatch.setattr(neg, "ollama", lambda *a, **k: "Sure! We guarantee payment by tomorrow.")
        r = neg.handle_message(s, q, "hello there, what next", use_llm=True)
        turn = s["history"][-1]
        assert "guarantee" not in r["reply"] and turn["checks"] and not all(c["passed"] for c in turn["checks"])
        assert turn["template"] == r["reply"]


class TestPricingProvenance:
    def test_every_line_says_where_its_numbers_come_from(self):
        q = pr.price_claim([item("front bumper", "crack", "high"), item("car door")], SWIFT)
        by = {l["part"]: l["provenance"] for l in q["line_items"]}
        assert by["front_bumper"]["part_price"].startswith("OBSERVED") and by["front_bumper"]["depreciation"].startswith("RULE")
        assert by["car_door"]["part_price"].startswith("N/A") and by["car_door"]["labour"].startswith("IMPUTED")
        w = pr.price_claim([item("front bumper", "crack", "high")], pr.DEMO_POLICIES["UP40K8060"])
        assert "secondary" in w["line_items"][0]["provenance"]["part_price"]

    def test_lesser_of_prefers_cheaper_replacement(self, monkeypatch):
        cheap = {**pr.TIERS["economy"], "labour": 6000, "parts": {**pr.TIERS["economy"]["parts"], "fender": 500}}
        monkeypatch.setitem(pr.TIERS, "economy", cheap)
        q = pr.price_claim([item("fender", "dent", "medium")], {**SWIFT, "model": "Alto"})
        l = q["line_items"][0]
        assert l["method"] == "REPLACE" and l["lesser_of"]["instead_of"] == "DENT_REPAIR"
        assert l["lesser_of"]["replace_net"] < l["lesser_of"]["repair_net"]


class TestSamplesAndRunner:
    def test_recorded_sample_is_cloned_and_runs_live(self, demo_api, monkeypatch, tmp_path):
        import subprocess
        import sys
        client, srv = demo_api
        h = demo_token(client)
        cid = upload(client, h, img=jpeg(320)).json()["claim_id"]
        root = tmp_path / "repo"
        (root / "tools").mkdir(parents=True)
        import shutil
        shutil.copy(srv.Path(srv.__file__).with_name("tools") / "save_demo_sample.py", root / "tools")
        r = subprocess.run([sys.executable, str(root / "tools" / "save_demo_sample.py"), cid, "door_dent", "Door dent", "test"],
                           env={**__import__("os").environ, "AUTOSETTLE_CLAIMS_DIR": str(srv.CLAIMS_DIR)},
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        monkeypatch.setattr(srv, "SAMPLES_DIR", root / "demo_samples")
        h2 = demo_token(client)
        lst = client.get("/api/demo_samples", headers=h2).json()
        assert [s["name"] for s in lst] == ["door_dent"] and lst[0]["title"] == "Door dent"
        c = client.post("/api/demo_sample", headers=h2, json={"name": "door_dent"}).json()
        assert c["claim_id"] != cid and c["recorded_sample"]["name"] == "door_dent"
        assert c["quote"]["line_items"] and c["chat"]["messages"]
        r = client.post("/api/chat_negotiate", headers=h2, json={"claim_id": c["claim_id"], "message": "how was it calculated"})
        assert r.status_code == 200
        assert client.get(f"/api/claim/{c['claim_id']}", headers=h).status_code == 404     # owned by the opener only
        assert client.post("/api/demo_sample", headers=h2, json={"name": "../etc"}).status_code == 404

    def test_samples_hidden_outside_demo(self, api):
        client, h, _ = api
        assert client.get("/api/demo_samples", headers=h).status_code == 404

    def test_redteam_runner_runs_the_real_suite(self, api):
        client, h, srv = api
        assert client.post("/api/admin/redteam_run", headers=h).status_code == 401          # customer token
        tok = client.post("/api/admin/login", json={"username": "admin", "password": "test-pass"}).json()["token"]
        r = client.post("/api/admin/redteam_run", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["total"] >= 60 and j["held"] == j["total"]
        assert any(x["expected"] for x in j["results"])
        assert client.get("/api/admin/redteam_last", headers={"Authorization": f"Bearer {tok}"}).json()["total"] == j["total"]
