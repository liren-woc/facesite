# Hairstyle References

This directory now contains the first China-first vetted reference set.

Current v1 coverage:

- `cn_natural_side_part`
- `cn_short_textured_crop`
- `cn_full_bangs_bob`
- `cn_soft_face_framing`

Only add or keep references that meet all of the following:

- real natural-color portraits
- front-view or near-front-view hair layout suitable for front try-on
- visually matches the catalog label
- Chinese mainstream aesthetic first, Korean support second
- no heavy accessories, extreme makeup, fantasy colors, or celebrity-event styling

Recommended workflow:

1. download the raw source image into the style directory as `front.jpg`
2. run `python scripts/prepare_reference_library.py`
3. run `python scripts/check_reference_library.py`
4. only keep entries that pass the validation report
