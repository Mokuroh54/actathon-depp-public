// ============================================================
//  Logitech C270 minimal clip plate  (~5 g, ~10 min print)
//  ------------------------------------------------------------
//  Print flat. No supports. Use 0.28 mm layer height for speed.
//
//  The camera's foot rests on the plate; its fold-down flap
//  drops into the slot in the ridge at the back, which locks
//  the camera against sliding. A zip tie through the two side
//  slots and over the foot stops it lifting out.
//
//  To raise it: zip-tie or VHB-tape the plate to a wooden dowel,
//  pencil, ruler, or whatever you have. Or tape it straight to
//  the robot roof.
// ============================================================

/* [Measured from your C270 clip flap] */
flap_w        = 40.3;
flap_h        = 18.9;
flap_t        = 2.3;
fit_clearance = 0.6;   // 0.4 tight .. 0.8 loose

/* [Plate] */
plate_t       = 1.6;   // 8 layers at 0.2 mm
plate_front   = 30;    // flat area in front of the slot for the foot
wall          = 1.6;   // ridge wall thickness (4 perimeters)
slot_depth    = 9;     // how far the flap goes in; full flap is 18.9, 9 is plenty
tie_slot_w    = 6;
tie_slot_l    = 2.4;

/* [Hidden] */
$fn = 24;
eps = 0.01;
slot_w = flap_w + fit_clearance;
slot_t = flap_t + fit_clearance;
plate_w = slot_w + 2*wall;
ridge_d = slot_t + 2*wall;
plate_d = plate_front + ridge_d;
ridge_h = plate_t + slot_depth;

difference() {
    union() {
        // plate
        cube([plate_w, plate_d, plate_t]);
        // ridge along the back edge
        translate([0, plate_d - ridge_d, 0]) cube([plate_w, ridge_d, ridge_h]);
    }
    // flap slot (floor = plate thickness)
    translate([wall, plate_d - ridge_d + wall, plate_t]) cube([slot_w, slot_t, ridge_h]);
    // zip-tie slots: pair for a tie over the foot (side to side)
    for (x = [wall + 1, plate_w - wall - 1 - tie_slot_l])
        translate([x, plate_front/2 - tie_slot_w/2, -1]) cube([tie_slot_l, tie_slot_w, plate_t + 2]);
    // pair for strapping the plate to a stick (front to back)
    for (y = [4, plate_front - 4 - tie_slot_l])
        translate([plate_w/2 - tie_slot_w/2, y, -1]) cube([tie_slot_w, tie_slot_l, plate_t + 2]);
}
