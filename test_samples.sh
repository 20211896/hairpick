#!/bin/bash

API_URL="http://127.0.0.1:8001/api/analyze-face"

for f in $(ls sample* 2>/dev/null | sort -V); do
  echo "========== $f =========="
  curl -s -X POST "$API_URL" -F "file=@$f" | python3 -c '
import sys, json

try:
    d = json.load(sys.stdin)
except Exception as e:
    print("JSON parse error:", e)
    sys.exit(1)

face_shape = d.get("face_shape", {})
ratios = d.get("ratios", {})

label = face_shape.get("label_ko")
shape_type = face_shape.get("type")

print("face_shape:", label, "(" + str(shape_type) + ")")
print("confidence:", face_shape.get("confidence"))
print("second:", face_shape.get("second_candidate"))
print("scores:", face_shape.get("score_breakdown"))
print("L/W:", ratios.get("face_length_width_ratio"))
print("F/C:", ratios.get("forehead_to_cheekbone_ratio"))
print("J/C:", ratios.get("jaw_to_cheekbone_ratio"))
print("LowerJaw/C:", ratios.get("lower_jaw_width_to_cheekbone_ratio"))
print("JawTaper:", ratios.get("jaw_taper_ratio"))
print("LowerFace/H:", ratios.get("lower_face_height_to_face_height_ratio"))
'
  echo
done
