"""Scan a 3D object with the eye, and print it the moment the scan ends.

xr_session.py --print-port <COMn | auto | sim> turns a 3D-object session into
scan-and-print:

  PRE-CONNECT (before the EEG recorder starts, before anything is shown)
    1. the object is one the printing step can make (the die: die_print.py)
    2. the operator has said the print bed is clear (--print-bed-clear;
       quest_run.ps1 -PrintPort asks)
    3. the slicer works: the die about to be shown is built and sliced now,
       with the same settings the decoded one will get, which also gives
       the print time and filament
    4. the printer answers: firmware and temperatures read, and the
       connection is held open with a status poll every few seconds for the
       whole scan; a printer switched off mid-scan is reported at once
    Any failure stops the session here, before the subject has scanned a
    single cell.

  PRINT (the moment the decoded object is painted back on the mesh)
    the decoded die is built (STL, relief: black cells sunk, red raised) and
    sliced, the held connection is handed to a print process of its own
    (printer.py print), and the print starts. It runs to the end whether or
    not the session is still open; the session shows its progress while it
    lingers. runs/<session>/print/ holds the STL, G-code, print_log.txt and
    print_status.json; `python printer.py cancel runs/<session>/print` stops
    it with the heaters off.
"""
import json
import os
import time

import numpy as np

import printer

HERE = os.path.dirname(os.path.abspath(__file__))
PRINTABLE = {"die"}                      # objects die_print.py can make


def mesh_spec(a):
    """The 3D object this session scans, or None."""
    if getattr(a, "mesh_object", None):
        return a.mesh_object
    import mesh3d
    if a.mode == "bwb" and mesh3d.is_mesh(getattr(a, "bwb_target", "")):
        return a.bwb_target
    return None


def prepare_planes_target(a, session):
    """--mode planes --mesh-object: the object's atlas becomes the colour
    target (one pixel a cell) and sets the grid size."""
    import mesh3d
    import mesh_planes
    obj = mesh3d.from_spec(a.mesh_object, a.mesh_layout)
    if not obj.mask.all():
        raise ValueError(f"{obj.name} in the {obj.layout} layout has holes; the planes scan "
                         "visits every cell (use --mesh-layout atlas)")
    path = os.path.join(session, "mesh_atlas.png")
    mesh_planes.atlas_png(obj, path)
    a.color_target, a.color_grid_w, a.color_grid_h = path, obj.grid_w, obj.grid_h
    print(f"3D object {obj.name}: {len(obj.cells)} cells, atlas {obj.grid_w} x {obj.grid_h} "
          f"-> {path}")
    return obj


def paint_planes(a, session):
    """After the planes decode: the decoded grid painted back on the object
    (mesh_planes.py paint)."""
    import argparse
    import mesh_planes
    mesh_planes.cmd_paint(argparse.Namespace(session=session, object=a.mesh_object,
                                             layout=a.mesh_layout))
    with open(os.path.join(session, "mesh_planes_result.json")) as f:
        return json.load(f)


class PrintJob:
    def __init__(self, a, session, log=print):
        self.a, self.session, self.log = a, session, log
        self.folder = os.path.join(session, "print")
        self.port = a.print_port
        self.link = None
        self.ready = None              # pre-connect summary
        self.launched = None           # the print process, once started
        self.lost = None

    # ------------------------------------------------ pre-connect
    def preflight(self):
        a, log = self.a, self.log
        log("PRINTER PRE-CONNECT: the object, the bed, the slicer and the printer, "
            "before any scanning")
        spec = mesh_spec(a)
        if spec is None:
            raise RuntimeError("--print-port needs a 3D object to print: --mode planes --mesh-object "
                               "mesh:die, or --mode bwb --bwb-target mesh:die")
        import mesh3d
        obj = mesh3d.from_spec(spec, a.mesh_layout)
        if obj.name not in PRINTABLE or (obj.grid_w, obj.grid_h) != (9, 6):
            raise RuntimeError(f"printing is ready for the die (mesh:die, 9 x 6 atlas); "
                               f"{obj.name} cannot be printed yet")
        log(f"  1. object: {obj.name}, {len(obj.cells)} cells")
        if not a.print_bed_clear:
            raise RuntimeError("the printer will start on its own when the scan ends: clear the "
                               "print bed, load filament, then add --print-bed-clear "
                               "(quest_run.ps1 -PrintPort asks for this)")
        log("  2. bed: the operator says it is clear and filament is loaded")
        # 3. the slicer, on the object that is about to be shown
        import die_print
        if die_print.find_cura() is None:
            raise RuntimeError("no slicer: UltiMaker Cura is not installed on this PC (it brings "
                               "CuraEngine, which die_print.py slices with). Install it from "
                               "ultimaker.com, or set CURA_DIR to its folder")
        dry = os.path.join(self.folder, "preflight")
        os.makedirs(dry, exist_ok=True)
        names, rgb = __import__("bwb").classes_of(obj.colours)
        shown = rgb[obj.target()]
        mesh3d.write_obj(obj, shown, os.path.join(dry, "mesh_shown.obj"))
        t0 = time.time()
        res = die_print.make_print(dry, mesh="mesh_shown.obj", size=a.print_size, depth=a.print_depth,
                                   name="die_shown", out=dry, slice_models=(a.print_model,),
                                   printer=a.printer, extra=self.extra(), log=lambda *x: None)
        info = res["info"][a.print_model] or {}
        est = float(info.get("TIME", 0) or 0)
        log(f"  3. slicer: CuraEngine in {die_print.CURA_DIR}, profile {a.printer}; the shown die "
            f"slices in {time.time() - t0:.0f} s: about {est / 3600:.1f} h, "
            f"{info.get('Filament used', '?')} of filament")
        # 4. the printer
        self.link = printer.connect(self.port, log=lambda *x: log("    " + " ".join(map(str, x))))
        self.port = self.link.port
        log(f"  4. printer: {self.link.describe()}")
        hot = self.link.temps.get("T", (0, 0))[0]
        if hot > 60:
            log(f"     note: the hot end is at {hot:.0f} C already (a print just finished?)")

        def lost(why):
            self.lost = why
            log(f"PRINTER LOST during the scan ({why}): the print will try to reconnect "
                "when the scan ends; check the printer's power and USB cable")
        self.link.start_keepalive(every=5.0, on_lost=lost)
        self.ready = dict(object=obj.name, port=self.port, firmware=self.link.firmware,
                          estimate_s=est, filament=info.get("Filament used"),
                          model=a.print_model, size_mm=a.print_size)
        with open(os.path.join(self.folder, "preconnect.json"), "w") as f:
            json.dump(self.ready, f, indent=1)
        log(f"PRINTER READY on {self.port}: the decoded {obj.name} prints as soon as the scan "
            f"ends (about {est / 3600:.1f} h)")
        return self.ready

    def extra(self):
        return dict(s.split("=", 1) for s in (self.a.print_set or []))

    def headline(self):
        if self.lost:
            return f"printer LOST ({self.port}) - check it"
        return (f"printer ready on {self.port}: prints when the scan ends"
                if self.ready else "")

    # ------------------------------------------------ print
    def release(self):
        if self.link is not None:
            self.link.close()
            self.link = None

    def finish(self, decoded_ok, why=""):
        """Build, slice and start printing the decoded object. Returns a dict
        describing the started print, or None."""
        log, a = self.log, self.a
        self.release()                          # the print process takes the port over
        if not decoded_ok:
            log(f"NOT PRINTING: the scan was not decoded ({why})")
            return None
        if not os.path.exists(os.path.join(self.session, "mesh_decoded.obj")):
            log("NOT PRINTING: no mesh_decoded.obj in the session")
            return None
        import die_print
        t0 = time.time()
        log("building and slicing the decoded die...")
        res = die_print.make_print(self.session, mesh="mesh_decoded.obj", size=a.print_size,
                                   depth=a.print_depth, name="die_decoded_eeg", out=self.folder,
                                   slice_models=(a.print_model,), printer=a.printer,
                                   extra=self.extra(), log=lambda *x: None)
        gcode = res["gcode"][a.print_model]
        proc, logf = printer.start_print_process(gcode, self.port, self.folder, a.print_baud)
        info = res["info"][a.print_model] or {}
        self.launched = dict(gcode=gcode, pid=proc.pid, log=logf, port=self.port,
                             cells=res["cells"], seconds_to_start=round(time.time() - t0, 1),
                             estimate_s=float(info.get("TIME", 0) or 0))
        log(f"PRINTING: {os.path.basename(gcode)} to {self.port} (print process {proc.pid}, "
            f"{self.launched['seconds_to_start']:.0f} s after the scan was decoded; about "
            f"{self.launched['estimate_s'] / 3600:.1f} h). Progress: python printer.py status "
            f"{self.folder}   stop: python printer.py cancel {self.folder}")
        return self.launched

    def status_text(self):
        path = os.path.join(self.folder, printer.STATUS)
        if not os.path.exists(path):
            return "printer connecting..."
        try:
            s = json.load(open(path))
        except (OSError, ValueError):
            return ""
        t = s.get("temps") or {}
        temps = " ".join(f"{'nozzle' if k == 'T' else 'bed'} {v[0]:.0f}/{v[1]:.0f} C" for k, v in
                         sorted(t.items(), reverse=True))
        return (f"PRINTING {100 * s.get('progress', 0):.0f}% - layer {s.get('layer')} of "
                f"{s.get('layers')} - {temps}" if s.get("state") == "printing"
                else f"print {s.get('state')}" + (f": {s['error']}" if s.get("error") else ""))
