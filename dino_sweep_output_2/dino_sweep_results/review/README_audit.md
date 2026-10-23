# DINO configuration sweep results

Five configurations were run on the same photos using one loaded Grounding DINO model. The model was not fine-tuned; this is an inference/prompt/settings comparison.

## Configurations

1. `01_baseline`: current thresholds and prompt (box .35, text .25, NMS .50, cross-label NMS .80).
2. `02_recall`: lower detection thresholds (.30/.20); can increase false detections.
3. `03_precision`: higher box threshold (.40/.25); can miss small or subtle parts.
4. `04_keep_overlaps`: cross-label NMS .95 to reduce suppression of overlapping labels; may retain duplicates.
5. `05_context_words`: adds front/rear/bonnet/side-door wording for commonly confused parts.

## How to audit strictly

- Review each original alongside that configuration's annotated image. The original photos are included under `_extracted_photos` in the output folder (the ZIP includes them for reproducibility).
- In `detection_review.csv`, mark label correctness and box localization separately. A box can be in the right place but have the wrong part label.
- In `image_review.csv`, record missed parts as well as false or wrongly labelled detections. Do not mark an image fully correct just because its visible boxes look plausible; inspect the whole photo for missed parts.
- For strict image-level correctness, mark `all_material_detections_correct_YN=Y` only when all material detections have the right part and location and no material false detection is present. Record missed real parts separately; any material miss should also make the image-level result N.
- A higher number of detections or higher average confidence does not mean better performance. Choose based on the same human labels across all configurations, then examine confusion pairs such as hood/door and headlight/taillight.

The CSV starts blank by design. This runner cannot infer ground truth from DINO's own output. Do not report the config summary's box counts or confidence as accuracy.
