// ============================================================
//  Logitech C270 mast mount  (v4: thin, prints lying flat)
//  ------------------------------------------------------------
//  Loads already in print orientation. No supports.
//  X = up the mast, Y = side to side, Z = "up" on the printer.
//
//  S-shape: base plate at one end, camera plate at the other, thin
//  mast between them with a rib along each edge.
//  The camera foot rests on the camera plate. Its fold-down flap
//  drops THROUGH a slot cut in the camera plate and lies flat
//  against the mast, between the ribs. That locks it in every
//  direction except straight up; a zip tie over the foot does that.
//  Camera lens faces +Z (the direction the plates stick out).
// ============================================================

/* [Measured from your C270 clip flap] */
flap_w        = 40.3;
flap_h        = 18.9;
flap_t        = 2.3;
fit_clearance = 0.6;

/* [Sizes] */
mast_h        = 120;   // base plate top -> camera plate top
mast_w        = 46;    // mast width; flap must fit between the ribs
mast_t        = 3;     // mast thickness
rib_h         = 5;     // stiffening ribs along both edges (0 = none)
rib_t         = 2.4;
base_w        = 50;    // base plate, side to side
base_d        = 35;    // base plate, front to back
base_t        = 2.4;
gusset        = 15;    // in-plane fins at the base
cam_plate_t   = 2.4;   // camera plate thickness
cam_front     = 22;    // flat area for the foot, in front of the flap slot
tie_w         = 5;  tie_t = 2.4;
strap_w       = 14; strap_t = 3.2;

/* [Hidden] */
eps = 0.01;
slot_w = flap_w + fit_clearance;
slot_t = flap_t + fit_clearance;
cam_w  = mast_w;
cam_d  = mast_t + slot_t + cam_front;  // camera plate depth (front to back)
x_top  = base_t + mast_h;            // camera plate top surface

difference() {
    union() {
        // base plate (stands on edge during print)
        translate([0, -base_w/2, 0]) cube([base_t, base_w, base_d]);
        // mast plate + ribs
        translate([base_t - eps, -mast_w/2, 0]) cube([mast_h + 2*eps, mast_w, mast_t]);
        if (rib_h > 0) for (y = [-mast_w/2, mast_w/2 - rib_t])
            translate([base_t - eps, y, 0]) cube([mast_h - cam_plate_t + 2*eps, rib_t, mast_t + rib_h]);
        // in-plane gussets at the base
        for (s = [-1, 1]) translate([base_t - eps, 0, 0]) linear_extrude(mast_t)
            polygon([[0, s*mast_w/2], [gusset, s*mast_w/2], [0, s*base_w/2]]);
        // camera plate (stands on edge), extends forward (+Z) from the mast
        translate([x_top - cam_plate_t, -cam_w/2, 0]) cube([cam_plate_t, cam_w, cam_d]);
    }
    // flap slot straight through the camera plate, just in front of the mast plate
    translate([x_top - cam_plate_t - 1, -slot_w/2, mast_t])
        cube([cam_plate_t + 2, slot_w, slot_t]);
    // zip-tie holes through the camera plate, either side of the foot
    for (y = [-1, 1])
        translate([x_top - cam_plate_t - 1, y*(cam_w/2 - 3.5) - tie_t/2, mast_t + slot_t + cam_front/2 - tie_w/2])
            cube([cam_plate_t + 2, tie_t, tie_w]);
    // strap slots in the base
    for (y = [-1, 1])
        translate([-1, y*(base_w/2 - 6) - strap_t/2, base_d/2 - strap_w/2]) cube([base_t + 2, strap_t, strap_w]);
    translate([-1, -strap_w/2, base_d - 5 - strap_t]) cube([base_t + 2, strap_w, strap_t]);
}
