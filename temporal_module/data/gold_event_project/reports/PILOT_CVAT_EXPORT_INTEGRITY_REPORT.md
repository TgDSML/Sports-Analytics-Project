# Pilot CVAT Export Integrity Report

## 1. Pilot ZIP Inventory

- Total `cvat_export__*.zip` count: 3
- Total ZIP files in export folder: 3
- Exactly three pilot exports present: yes

- `cvat_export__england_epl__2015_2016__2015_08_30___18_00_Swansea_2___1_Manchester_United__h1_720p.zip`: 103593 bytes
- `cvat_export__england_epl__2015_2016__2015_08_30___18_00_Swansea_2___1_Manchester_United__h2_720p.zip`: 97326 bytes
- `cvat_export__england_epl__2015_2016__2015_09_26___17_00_Manchester_United_3___0_Sunderland__h1_720p.zip`: 92969 bytes
- Invalid ZIP names: none
- Naming convention violations: none

## 2. Manifest Mapping Status

- `england_epl__2015_2016__2015_08_30___18_00_Swansea_2___1_Manchester_United__h1_720p`: ok
  - Expected video filename: `england_epl__2015_2016__2015_08_30___18_00_Swansea_2___1_Manchester_United__h1_720p.mp4`
  - Expected CVAT task name: `gold_v1__england_epl__2015_2016__2015_08_30___18_00_Swansea_2___1_Manchester_United__h1_720p.mp4`
- `england_epl__2015_2016__2015_08_30___18_00_Swansea_2___1_Manchester_United__h2_720p`: ok
  - Expected video filename: `england_epl__2015_2016__2015_08_30___18_00_Swansea_2___1_Manchester_United__h2_720p.mp4`
  - Expected CVAT task name: `gold_v1__england_epl__2015_2016__2015_08_30___18_00_Swansea_2___1_Manchester_United__h2_720p.mp4`
- `england_epl__2015_2016__2015_09_26___17_00_Manchester_United_3___0_Sunderland__h1_720p`: ok
  - Expected video filename: `england_epl__2015_2016__2015_09_26___17_00_Manchester_United_3___0_Sunderland__h1_720p.mp4`
  - Expected CVAT task name: `gold_v1__england_epl__2015_2016__2015_09_26___17_00_Manchester_United_3___0_Sunderland__h1_720p.mp4`

## 3. ZIP Readability And Native CVAT Structure

- `cvat_export__england_epl__2015_2016__2015_08_30___18_00_Swansea_2___1_Manchester_United__h1_720p.zip`
  - Readable/not corrupted: yes
  - Annotation XML: `annotations.xml`
  - Native CVAT annotations structure: yes
  - Task name in XML: `gold_v1__england_epl__2015_2016__2015_08_30___18_00_Swansea_2___1_Manchester_United__h1_720p.mp4`
  - Archive members:
    - `annotations.xml`
- `cvat_export__england_epl__2015_2016__2015_08_30___18_00_Swansea_2___1_Manchester_United__h2_720p.zip`
  - Readable/not corrupted: yes
  - Annotation XML: `annotations.xml`
  - Native CVAT annotations structure: yes
  - Task name in XML: `gold_v1__england_epl__2015_2016__2015_08_30___18_00_Swansea_2___1_Manchester_United__h2_720p.mp4`
  - Archive members:
    - `annotations.xml`
- `cvat_export__england_epl__2015_2016__2015_09_26___17_00_Manchester_United_3___0_Sunderland__h1_720p.zip`
  - Readable/not corrupted: yes
  - Annotation XML: `annotations.xml`
  - Native CVAT annotations structure: yes
  - Task name in XML: `gold_v1__england_epl__2015_2016__2015_09_26___17_00_Manchester_United_3___0_Sunderland__h1_720p.mp4`
  - Archive members:
    - `annotations.xml`

## 4. Event Track Summary By Clip

### `england_epl__2015_2016__2015_08_30___18_00_Swansea_2___1_Manchester_United__h1_720p`

- Track 1: label `uncertain`, active `0`-`19`, first outside after `20`, inferred inclusive interval `0`-`19`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 2: label `pass`, active `49`-`95`, first outside after `96`, inferred inclusive interval `49`-`95`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 3: label `turnover`, active `101`-`121`, first outside after `122`, inferred inclusive interval `101`-`121`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 4: label `pass`, active `178`-`197`, first outside after `198`, inferred inclusive interval `178`-`197`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 5: label `pass`, active `236`-`255`, first outside after `256`, inferred inclusive interval `236`-`255`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 6: label `pass`, active `285`-`327`, first outside after `328`, inferred inclusive interval `285`-`327`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 7: label `pass`, active `341`-`365`, first outside after `366`, inferred inclusive interval `341`-`365`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 8: label `pass`, active `200`-`234`, first outside after `235`, inferred inclusive interval `200`-`234`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 9: label `shot`, active `367`-`380`, first outside after `381`, inferred inclusive interval `367`-`380`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 10: label `uncertain`, active `390`-`427`, first outside after `428`, inferred inclusive interval `390`-`427`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 11: label `uncertain`, active `481`-`545`, first outside after `546`, inferred inclusive interval `481`-`545`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 12: label `uncertain`, active `550`-`572`, first outside after `573`, inferred inclusive interval `550`-`572`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 13: label `uncertain`, active `574`-`598`, first outside after `599`, inferred inclusive interval `574`-`598`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 14: label `pass`, active `600`-`612`, first outside after `613`, inferred inclusive interval `600`-`612`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 15: label `uncertain`, active `628`-`688`, first outside after `689`, inferred inclusive interval `628`-`688`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 16: label `pass`, active `690`-`720`, first outside after `721`, inferred inclusive interval `690`-`720`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 17: label `uncertain`, active `722`-`747`, first outside after `748`, inferred inclusive interval `722`-`747`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
### `england_epl__2015_2016__2015_08_30___18_00_Swansea_2___1_Manchester_United__h2_720p`

- Track 1: label `uncertain`, active `13`-`49`, first outside after `50`, inferred inclusive interval `13`-`49`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 2: label `pass`, active `51`-`109`, first outside after `110`, inferred inclusive interval `51`-`109`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 3: label `pass`, active `112`-`149`, first outside after `150`, inferred inclusive interval `112`-`149`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 4: label `pass`, active `240`-`293`, first outside after `294`, inferred inclusive interval `240`-`293`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 5: label `uncertain`, active `299`-`497`, first outside after `498`, inferred inclusive interval `299`-`497`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 6: label `pass`, active `518`-`601`, first outside after `602`, inferred inclusive interval `518`-`601`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 7: label `pass`, active `604`-`635`, first outside after `636`, inferred inclusive interval `604`-`635`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
### `england_epl__2015_2016__2015_09_26___17_00_Manchester_United_3___0_Sunderland__h1_720p`

- Track 1: label `uncertain`, active `0`-`70`, first outside after `71`, inferred inclusive interval `0`-`70`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 2: label `uncertain`, active `243`-`261`, first outside after `262`, inferred inclusive interval `243`-`261`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 3: label `pass`, active `265`-`287`, first outside after `288`, inferred inclusive interval `265`-`287`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 4: label `pass`, active `304`-`341`, first outside after `342`, inferred inclusive interval `304`-`341`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 5: label `pass`, active `344`-`403`, first outside after `404`, inferred inclusive interval `344`-`403`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 6: label `carry`, active `416`-`480`, first outside after `481`, inferred inclusive interval `416`-`480`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 7: label `pass`, active `483`-`524`, first outside after `525`, inferred inclusive interval `483`-`524`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 8: label `carry`, active `531`-`561`, first outside after `562`, inferred inclusive interval `531`-`561`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 9: label `pass`, active `570`-`627`, first outside after `628`, inferred inclusive interval `570`-`627`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 10: label `turnover`, active `654`-`664`, first outside after `665`, inferred inclusive interval `654`-`664`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 11: label `pass`, active `689`-`743`, first outside after `744`, inferred inclusive interval `689`-`743`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes
- Track 12: label `pass`, active `342`-`342`, first outside after `343`, inferred inclusive interval `342`-`342`, outside state available: True, geometry fields present but unnecessary: `xbr,xtl,ybr,ytl`, unambiguous: yes

## 5. Event Counts By Class

- carry: 2
- pass: 20
- turnover: 2
- shot: 1
- uncertain: 11
- Total valid tracks: 36
- Total invalid tracks: 0

## 6. Frame-Range, Duplicate, And Overlap Validation

- `england_epl__2015_2016__2015_08_30___18_00_Swansea_2___1_Manchester_United__h1_720p`: video frames `750`, temporal frames `0`-`749` (`750` rows)
- `england_epl__2015_2016__2015_08_30___18_00_Swansea_2___1_Manchester_United__h2_720p`: video frames `750`, temporal frames `0`-`749` (`748` rows)
- `england_epl__2015_2016__2015_09_26___17_00_Manchester_United_3___0_Sunderland__h1_720p`: video frames `750`, temporal frames `0`-`749` (`748` rows)
- Duplicate interval violations: 0
- Overlap violations: 0
- Unknown-label tracks: 0
- Uncertain events: 11
- All annotation tracks can be represented as `clip_id,event_type,start_frame,end_frame`: yes

## 7. Importer Compatibility Verdict

- Discovers ZIPs from expected export folder: `verified`. Uses DEFAULT_GOLD_ROOT/cvat_exports and glob("cvat_export__*.zip").
- Maps ZIP filenames to clip IDs exactly: `verified`. Derives clip_id from exact cvat_export__ prefix and rejects unknown clips.
- Parses native CVAT export structure found in pilot ZIPs: `verified`. Finds XML and parses CVAT annotations; pilot ZIPs contain native annotations.xml.
- Uses active frames / outside state for interval extraction: `implemented_differently`. active_intervals ignores boxes with outside=1 and groups non-outside frames; it does not explicitly use the outside frame to set end_frame.
- Ignores rectangle position and size: `verified`. Importer reads only track label, box frame, and outside attribute.
- Uses inclusive end-frame semantics: `verified`. End frame is the last active/non-outside frame.
- Rejects unknown labels: `implemented_differently`. Importer silently skips unknown labels rather than recording an error.
- Detects overlapping events: `verified`. Adds validation errors for overlapping intervals.
- Handles the final active frame correctly: `verified`. Always appends the final active run after iteration.

Compatibility conclusion: the current importer can parse these pilot ZIPs without code changes if the observed dense active-frame CVAT box entries are preserved. It ignores rectangle geometry and uses non-outside frame entries as inclusive intervals. Unknown labels are skipped rather than rejected with an error, which is implemented differently from a strict reject policy but does not affect these pilots because no unknown labels were found.

## 8. Can The Rectangle Stay Fixed Anywhere In The Frame?

Yes. The rectangle marker can stay fixed anywhere in the frame for these event annotations. The importer reads the track label, `frame`, and `outside` state only; it does not read `xtl`, `ytl`, `xbr`, `ybr`, or `points` for interval extraction. Geometry is therefore unnecessary for deriving `clip_id,event_type,start_frame,end_frame`.

## 9. Final Result

PILOT EXPORTS SAFE TO IMPORT
