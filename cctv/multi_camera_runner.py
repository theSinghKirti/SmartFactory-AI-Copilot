import logging
from database.db import get_active_cameras
from cctv.safety_engine import check, get_model
from cctv.tracker import track
from cctv.violation_tracker import ViolationTracker
from cctv.incident_pipeline import handle_violations
from cctv.stream import get_frames

logger = logging.getLogger(__name__)

# Configurable frame processing throttle limit per camera pass sweep
MAX_FRAMES_PER_CAMERA = 30

def run_all_cameras(progress_callback=None, run_id=None):
    """
    Process all active registered cameras from the SQLite cameras table,
    running safety inference and routing tracked violations with the
    active session's run_id identity.

    Camera sources are read from the database (registered via db.add_camera()
    or seeded by db.init_db()), not from a hardcoded list.
    """
    cameras = get_active_cameras()

    if not cameras:
        logger.warning(
            "No active cameras found in the database. "
            "Register cameras using db.add_camera() or check that the video "
            "source files exist at the paths in the cameras table."
        )
        return

    yolo_model = get_model()

    for camera_row in cameras:
        # sqlite3.Row supports both index and name access
        camera_name = camera_row["name"]
        zone_name   = camera_row["zone"]
        source_path = camera_row["source_path"]

        # Independent tracker per camera — no trajectory leakage between zones
        tracker = ViolationTracker(timeout_seconds=3)

        try:
            frames = get_frames(source_path)

            for i, frame in enumerate(frames):
                if i >= MAX_FRAMES_PER_CAMERA:
                    break

                results = yolo_model.predict(source=frame, conf=0.35, verbose=False)
                tracked = track(results[0])
                violations = check(tracked, yolo_model.names)

                handle_violations(
                    violations=violations,
                    frame=frame,
                    tracker_instance=tracker,
                    camera=camera_name,
                    zone=zone_name,
                    is_last_frame=False,
                    run_id=run_id,
                )

            # Flush remaining open violations at stream boundary
            handle_violations(
                violations=[],
                frame=None,
                tracker_instance=tracker,
                camera=camera_name,
                zone=zone_name,
                is_last_frame=True,
                run_id=run_id,
            )

            if progress_callback:
                progress_callback(camera_name)

        except Exception as e:
            logger.error(
                "Failed processing camera '%s' (source: %s): %s",
                camera_name, source_path, e,
                exc_info=True,
            )
            # Continue with remaining cameras
            continue