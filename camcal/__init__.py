"""camcal: camera calibration for a Jetson's cameras (and any other OpenCV can open).

    python3 -m camcal board       a ChArUco board or a chessboard, as a PDF at its true size
    python3 -m camcal capture     pictures of the board, taken when a view adds something new
    python3 -m camcal monitor     a window here that follows a capture there: count, grade, flatness
    python3 -m camcal calibrate   intrinsics, with the model chosen on views held out, and a report
    python3 -m camcal validate    an existing calibration against pictures it has not seen
    python3 -m camcal undistort   rectified pictures, and the remap tables for an application
    python3 -m camcal extrinsics  the camera in the vehicle's frame, from a board on the ground
    python3 -m camcal budget      ADAS: the error of a measured range at each range, and its causes
    python3 -m camcal ground      ADAS: pixel to the point of the road, and back
    python3 -m camcal bev         ADAS: the road from above, with the remap tables
    python3 -m camcal transform   the same calibration for a rotated, scaled or cropped picture
    python3 -m camcal synth       rendered pictures of a board from a known camera (ground truth)

Needs Python 3.8+, OpenCV 4.7+ (JetPack 6 has 4.8) and NumPy; Matplotlib for the report's plots.
"""

__version__ = "1.0.0"
