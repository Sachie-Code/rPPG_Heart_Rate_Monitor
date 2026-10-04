import threading
import time
import cv2
import numpy as np

from signal_processing import (calculate_rppg, get_rppg_waveform)

class RPPGProcessor:
    def __init__(self):

        # Set the size of the video frame
        self.frame_width = 640
        self.frame_height = 480

        # Use OpenCV to detect the face
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        # Store the RGB values and their times
        self.rgb_values = []
        self.timestamps = []
        self.max_samples = 300

        # Store RGB values from the forehead and both cheeks
        self.forehead_rgb_values = []
        self.left_cheek_rgb_values = []
        self.right_cheek_rgb_values = []
        self.multi_roi_timestamps = []

        # Store the current heart rate and video speed
        self.bpm = 0.0
        self.fps = 0.0
        self.sample_fps = 0.0

        # Keep track of whether a face and ROI are detected
        self.face_detected = False
        self.roi_detected = False

        # Store the last detected face and ROI
        self.last_face = None
        self.last_roi = None

        # Count frames and control how often the face is detected
        self.frame_count = 0
        self.face_detection_interval = 5
        self.last_frame_time = time.time()

        # Allow a few missed face detections before resetting
        self.missed_face_count = 0
        self.max_missed_faces = 6

        # Protect shared data when multiple threads are running
        self.lock = threading.Lock()
        self.waveform_lock = threading.Lock()

        # Initial status of the heart-rate measurement
        self.bpm_status = "COLLECTING SIGNAL"
        self.waveform = None

        # Store the heatmap so it does not need to be rebuilt every frame
        self._heat_key = None
        self._heat_cache = None

        # Settings used to display text on the video
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale_title = 0.60
        self.font_scale_main = 0.55
        self.font_scale_small = 0.45

        self.font_thickness_title = 2
        self.font_thickness_main = 2
        self.font_thickness_small = 1

        self.font_line = cv2.LINE_AA

        # Start the thread that calculates the heart rate
        self.bpm_thread_running = True

        self.bpm_thread = threading.Thread(
            target=self.bpm_worker,
            daemon=True
        )

        self.bpm_thread.start()


    def stop(self):
        # Stop the heart-rate calculation thread
        self.bpm_thread_running = False


    def bpm_worker(self):
        # Keep calculating the heart rate while the program is running
        while self.bpm_thread_running:

            try:
                self.calculate_bpm()
                self.calculate_waveform()

            except Exception:
                with self.lock:
                    self.bpm_status = "SIGNAL PROCESSING"

            time.sleep(0.25)


    def _clear_signal_data(self):
        # Remove all stored signal data
        with self.lock:
            self.forehead_rgb_values.clear()
            self.left_cheek_rgb_values.clear()
            self.right_cheek_rgb_values.clear()

            self.multi_roi_timestamps.clear()

            self.rgb_values.clear()
            self.timestamps.clear()

            self.bpm = 0.0
            self.sample_fps = 0.0
            self.bpm_status = "COLLECTING SIGNAL"

        # Remove the old waveform
        with self.waveform_lock:
            self.waveform = None


    # Face Detection
    def _detect_face(self, frame):

        # Convert the image to grayscale
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Make the image smaller to detect the face faster
        small_gray = cv2.resize(
            gray,
            None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_AREA
        )

        # Find faces in the image
        faces = self.face_detector.detectMultiScale(
            small_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        # If no face is found, return nothing
        if len(faces) == 0:
            return None

        # Use the largest detected face
        face = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        x, y, w, h = face

        # Change the coordinates back to the original 640x480 size
        x = int(x * 2)
        y = int(y * 2)
        w = int(w * 2)
        h = int(h * 2)

        return x, y, w, h


    def _update_face(self, new_box):

        # Convert the new face position to numbers that can calculate with
        new_box = np.array(
            new_box,
            dtype=np.float64
        )

        if self.last_face is None:
            smooth_box = new_box

        else:
            # Get the previous face position
            old_box = np.array(
                self.last_face,
                dtype=np.float64
            )

            # Find the center of the old face
            old_center = (
                old_box[:2]
                + old_box[2:] / 2.0
            )

            # Find the center of the new face
            new_center = (
                new_box[:2]
                + new_box[2:] / 2.0
            )

            # Check how much the face moved
            movement = (
                np.linalg.norm(
                    new_center - old_center
                )
                / max(1.0, old_box[2])
            )

            if movement > 0.08:
                # If the face moved a lot, use the new position
                smooth_box = new_box

            else:
                # If the face moved only a little, smooth the position
                smooth_box = (
                    0.5 * old_box
                    + 0.5 * new_box
                )

        # Save the updated face position
        self.last_face = tuple(
            int(v)
            for v in smooth_box
        )

        self.face_detected = True
        self.missed_face_count = 0


    # Frame Processing
    def process_frame(
        self,
        frame,
        timestamp=None
    ):

        # Resize every frame to the same size
        frame = cv2.resize(
            frame,
            (
                self.frame_width,
                self.frame_height
            ),
            interpolation=cv2.INTER_AREA
        )

        # Use the given time, or use the current time
        if timestamp is None:
            timestamp = time.time()

        current_time = float(timestamp)

        # Count this frame
        self.frame_count += 1

        # Calculate the time since the previous frame
        delta = (
            current_time
            - self.last_frame_time
        )

        if delta > 0:

            # Calculate the current FPS
            current_fps = 1.0 / delta

            # Smooth the FPS value
            self.fps = (
                self.fps * 0.90
                + current_fps * 0.10
            )

        self.last_frame_time = current_time

        # Make a copy that will be displayed
        output = frame.copy()

        # Detect the face every few frames
        if (
            self.frame_count
            % self.face_detection_interval
            == 0
        ):

            detected_face = self._detect_face(
                frame
            )

            if detected_face is not None:

                # Update the face position
                self._update_face(
                    detected_face
                )

            else:

                # Count how many times the face was missed
                self.missed_face_count += 1

                if (
                    self.missed_face_count
                    >= self.max_missed_faces
                ):

                    # If the face is missing for too long, reset everything
                    self.face_detected = False
                    self.roi_detected = False
                    self.last_face = None
                    self.last_roi = None

                    self._clear_signal_data()


        # ROI Extraction
        if (
            self.face_detected
            and self.last_face is not None
        ):

            # Get the face position
            x, y, w, h = self.last_face

            # Make sure the face coordinates stay inside the image
            x = max(0, x)
            y = max(0, y)

            w = min(
                w,
                frame.shape[1] - x
            )

            h = min(
                h,
                frame.shape[0] - y
            )


            # Forehead
            # Select the forehead area
            roi_x1 = int(x + w * 0.25)
            roi_x2 = int(x + w * 0.75)

            roi_y1 = int(y + h * 0.12)
            roi_y2 = int(y + h * 0.32)

            forehead_roi = frame[
                roi_y1:roi_y2,
                roi_x1:roi_x2
            ]


            # Left cheek
            # Select the left cheek area
            left_cx = int(x + w * 0.30)
            left_cy = int(y + h * 0.60)

            left_rx = max(
                1,
                int(w * 0.16)
            )

            left_ry = max(
                1,
                int(h * 0.13)
            )

            left_x1 = max(
                0,
                left_cx - left_rx
            )

            left_x2 = min(
                frame.shape[1],
                left_cx + left_rx
            )

            left_y1 = max(
                0,
                left_cy - left_ry
            )

            left_y2 = min(
                frame.shape[0],
                left_cy + left_ry
            )

            left_cheek_roi = frame[
                left_y1:left_y2,
                left_x1:left_x2
            ]


            # Right cheek
            # Select the right cheek area
            right_cx = int(x + w * 0.70)
            right_cy = int(y + h * 0.60)

            right_rx = max(
                1,
                int(w * 0.16)
            )

            right_ry = max(
                1,
                int(h * 0.13)
            )

            right_x1 = max(
                0,
                right_cx - right_rx
            )

            right_x2 = min(
                frame.shape[1],
                right_cx + right_rx
            )

            right_y1 = max(
                0,
                right_cy - right_ry
            )

            right_y2 = min(
                frame.shape[0],
                right_cy + right_ry
            )

            right_cheek_roi = frame[
                right_y1:right_y2,
                right_x1:right_x2
            ]


            # Find the average RGB value of each face area
            if (
                forehead_roi.size > 0
                and left_cheek_roi.size > 0
                and right_cheek_roi.size > 0
            ):

                # Average color of the forehead
                forehead_mean = np.mean(
                    forehead_roi,
                    axis=(0, 1)
                )

                fb, fg, fr = forehead_mean

                # Average color of the left cheek
                left_mean = np.mean(
                    left_cheek_roi,
                    axis=(0, 1)
                )

                lb, lg, lr = left_mean

                # Average color of the right cheek
                right_mean = np.mean(
                    right_cheek_roi,
                    axis=(0, 1)
                )

                rb, rg, rr = right_mean


                with self.lock:

                    # Save the colors in RGB order
                    self.forehead_rgb_values.append(
                        [fr, fg, fb]
                    )

                    self.left_cheek_rgb_values.append(
                        [lr, lg, lb]
                    )

                    self.right_cheek_rgb_values.append(
                        [rr, rg, rb]
                    )

                    # Save the time of this measurement
                    self.multi_roi_timestamps.append(
                        current_time
                    )


                    # Create one RGB value by averaging the forehead and both cheeks
                    combined_rgb = (
                        np.array([fr, fg, fb])
                        + np.array([lr, lg, lb])
                        + np.array([rr, rg, rb])
                    ) / 3.0

                    self.rgb_values.append(
                        combined_rgb.tolist()
                    )

                    # Keep only the most recent samples
                    if (
                        len(self.forehead_rgb_values)
                        > self.max_samples
                    ):

                        self.forehead_rgb_values = (
                            self.forehead_rgb_values[
                                -self.max_samples:
                            ]
                        )

                        self.left_cheek_rgb_values = (
                            self.left_cheek_rgb_values[
                                -self.max_samples:
                            ]
                        )

                        self.right_cheek_rgb_values = (
                            self.right_cheek_rgb_values[
                                -self.max_samples:
                            ]
                        )

                        self.multi_roi_timestamps = (
                            self.multi_roi_timestamps[
                                -self.max_samples:
                            ]
                        )


                    if len(self.rgb_values) > self.max_samples:

                        self.rgb_values = (
                            self.rgb_values[
                                -self.max_samples:
                            ]
                        )

                        self.timestamps = (
                            self.timestamps[
                                -self.max_samples:
                            ]
                        )


                    # Save the current time
                    self.timestamps.append(
                        current_time
                    )

                    # Keep the number of timestamps limited
                    if len(self.timestamps) > self.max_samples:

                        self.timestamps = (
                            self.timestamps[
                                -self.max_samples:
                            ]
                        )


                # Save the forehead ROI position
                self.last_roi = (
                    roi_x1,
                    roi_y1,
                    roi_x2,
                    roi_y2
                )

                self.roi_detected = True

            else:
                self.roi_detected = False

        else:
            self.roi_detected = False


        # Visual Overlay
        # Draw the face information and heatmap on the frame
        if (
            self.face_detected
            and self.last_face is not None
            and self.last_roi is not None
        ):

            x, y, w, h = self.last_face

            rx1, ry1, rx2, ry2 = (
                self.last_roi
            )

            # Use the face and ROI positions as the heatmap key
            key = (
                x,
                y,
                w,
                h,
                rx1,
                ry1,
                rx2,
                ry2
            )

            # Build a new heatmap only when the position changes
            if key != self._heat_key:

                self._heat_cache = (
                    self._build_heat_overlay(
                        x,
                        y,
                        w,
                        h,
                        rx1,
                        ry1,
                        rx2,
                        ry2
                    )
                )

                self._heat_key = key


            if self._heat_cache is not None:

                (
                    bx1,
                    by1,
                    bx2,
                    by2,
                    inv_alpha,
                    heat_part
                ) = self._heat_cache

                # Mix the heatmap with the original image
                region = (
                    output[
                        by1:by2,
                        bx1:bx2
                    ].astype(np.float32)
                    * inv_alpha
                    + heat_part
                )

                output[
                    by1:by2,
                    bx1:bx2
                ] = np.clip(
                    region,
                    0,
                    255
                ).astype(np.uint8)


            # Draw the four corners around the face
            color = (235, 235, 235)
            thickness = 2

            length = max(
                20,
                int(min(w, h) * 0.12)
            )

            cv2.line(
                output,
                (x, y),
                (x + length, y),
                color,
                thickness,
                cv2.LINE_AA
            )

            cv2.line(
                output,
                (x, y),
                (x, y + length),
                color,
                thickness,
                cv2.LINE_AA
            )

            cv2.line(
                output,
                (x + w - length, y),
                (x + w, y),
                color,
                thickness,
                cv2.LINE_AA
            )

            cv2.line(
                output,
                (x + w, y),
                (x + w, y + length),
                color,
                thickness,
                cv2.LINE_AA
            )

            cv2.line(
                output,
                (x, y + h - length),
                (x, y + h),
                color,
                thickness,
                cv2.LINE_AA
            )

            cv2.line(
                output,
                (x, y + h),
                (x + length, y + h),
                color,
                thickness,
                cv2.LINE_AA
            )

            cv2.line(
                output,
                (x + w - length, y + h),
                (x + w, y + h),
                color,
                thickness,
                cv2.LINE_AA
            )

            cv2.line(
                output,
                (x + w, y + h - length),
                (x + w, y + h),
                color,
                thickness,
                cv2.LINE_AA
            )


        # Draw the information and waveform
        self.draw_overlay(output)
        self.draw_waveform(output)

        return output


    # Heatmap
    def _build_heat_overlay(
        self,
        x,
        y,
        w,
        h,
        rx1,
        ry1,
        rx2,
        ry2
    ):

        # Add some space around the face
        pad = 42

        bx1 = max(
            0,
            x - pad
        )

        by1 = max(
            0,
            y - pad
        )

        bx2 = min(
            self.frame_width,
            x + w + pad
        )

        by2 = min(
            self.frame_height,
            y + h + pad
        )

        # Make sure the area is valid
        if bx2 <= bx1 or by2 <= by1:
            return None

        small_h = by2 - by1
        small_w = bx2 - bx1

        # Create a mask for the face
        face_mask = np.zeros(
            (small_h, small_w),
            dtype=np.uint8
        )

        center = (
            int(x + w * 0.50) - bx1,
            int(y + h * 0.50) - by1
        )

        axes = (
            max(1, int(w * 0.46)),
            max(1, int(h * 0.50))
        )

        cv2.ellipse(
            face_mask,
            center,
            axes,
            0,
            0,
            360,
            255,
            -1
        )

        # Create a mask for the forehead ROI
        roi_mask = np.zeros_like(
            face_mask
        )

        cv2.rectangle(
            roi_mask,
            (
                rx1 - bx1,
                ry1 - by1
            ),
            (
                rx2 - bx1,
                ry2 - by1
            ),
            255,
            -1
        )

        # Create a mask for both cheeks
        cheek_mask = np.zeros_like(
            face_mask
        )

        cheek_axes = (
            max(1, int(w * 0.16)),
            max(1, int(h * 0.13))
        )

        left_center = (
            int(x + w * 0.30) - bx1,
            int(y + h * 0.60) - by1
        )

        right_center = (
            int(x + w * 0.70) - bx1,
            int(y + h * 0.60) - by1
        )

        cv2.ellipse(
            cheek_mask,
            left_center,
            cheek_axes,
            0,
            0,
            360,
            255,
            -1
        )

        cv2.ellipse(
            cheek_mask,
            right_center,
            cheek_axes,
            0,
            0,
            360,
            255,
            -1
        )

        # Keep the cheek areas inside the face
        cheek_mask = cv2.bitwise_and(
            cheek_mask,
            face_mask
        )

        # Combine the forehead and cheek areas
        combined = np.maximum(
            roi_mask,
            cheek_mask
        )

        # Make the edges softer
        soft = cv2.GaussianBlur(
            combined,
            (0, 0),
            14
        )

        # Create the heatmap
        heatmap = cv2.applyColorMap(
            soft,
            cv2.COLORMAP_MAGMA
        )

        # Control how strong the heatmap appears
        alpha = (
            soft.astype(np.float32)
            / 255.0
        ) * 0.55

        alpha = alpha[..., None]

        inv_alpha = 1.0 - alpha

        heat_part = (
            heatmap.astype(np.float32)
            * alpha
        )

        return (
            bx1,
            by1,
            bx2,
            by2,
            inv_alpha,
            heat_part
        )


    def draw_overlay(self, frame):

        # Create a small information panel
        panel = frame[8:92, 8:558]

        gray_box = np.full_like(
            panel,
            220
        )

        cv2.addWeighted(
            gray_box,
            0.30,
            panel,
            0.70,
            0,
            panel
        )

        # Display the title
        cv2.putText(
            frame,
            "rPPG HEART RATE MONITOR",
            (15, 23),
            self.font,
            0.45,
            (0, 0, 0),
            1,
            self.font_line
        )

        # Display the heart rate
        if self.bpm > 0:

            bpm_text = (
                f"HEART RATE  "
                f"{self.bpm:3.0f} BPM"
            )

        else:

            bpm_text = (
                "HEART RATE  --- BPM"
            )

        cv2.putText(
            frame,
            bpm_text,
            (15, 48),
            self.font,
            0.42,
            (0, 0, 0),
            1,
            self.font_line
        )


        # Draw a small heart symbol
        heart_x = 190
        heart_y = 38

        cv2.circle(
            frame,
            (heart_x + 5, heart_y),
            5,
            (0, 0, 255),
            -1,
            cv2.LINE_AA
        )

        cv2.circle(
            frame,
            (heart_x + 15, heart_y),
            5,
            (0, 0, 255),
            -1,
            cv2.LINE_AA
        )

        cv2.fillPoly(
            frame,
            [
                np.array(
                    [
                        [heart_x, heart_y + 2],
                        [heart_x + 20, heart_y + 2],
                        [heart_x + 10, heart_y + 15]
                    ],
                    dtype=np.int32
                )
            ],
            (0, 0, 255)
        )

        # Display the video FPS
        fps_text = (
            f"FPS      {self.fps:05.1f}"
        )

        cv2.putText(
            frame,
            fps_text,
            (15, 73),
            self.font,
            self.font_scale_small,
            (0, 0, 0),
            self.font_thickness_small,
            self.font_line
        )

        # Count how many samples have been collected
        with self.lock:
            samples = len(
                self.rgb_values
            )

        samples_text = (
            f"SAMPLES  {samples:03d}"
        )

        cv2.putText(
            frame,
            samples_text,
            (145, 73),
            self.font,
            self.font_scale_small,
            (0, 0, 0),
            self.font_thickness_small,
            self.font_line
        )

    # Waveform
    def calculate_waveform(self):

        # Get the latest RGB values
        with self.lock:

            if len(self.rgb_values) < 30:
                return

            values = np.array(
                self.rgb_values[-150:],
                dtype=np.float64
            )

            timestamps = np.array(
                self.timestamps[-len(values):],
                dtype=np.float64
            )

        # Make sure there are enough timestamps
        if len(timestamps) < 30:
            return

        # Make sure the timestamps are increasing
        if np.any(
            np.diff(timestamps) <= 0
        ):
            return

        # Calculate how long the signal lasted
        duration = (
            timestamps[-1]
            - timestamps[0]
        )

        if duration <= 0:
            return

        # Calculate the sampling speed
        sample_fps = (
            len(timestamps) - 1
        ) / duration

        # Create the rPPG waveform
        waveform = get_rppg_waveform(
            values,
            sample_fps
        )

        if waveform is None:
            return

        # Save the waveform
        with self.waveform_lock:
            self.waveform = (
                waveform.copy()
            )

    def draw_waveform(self, frame):

        # Get the latest waveform
        with self.waveform_lock:

            if self.waveform is None:
                return

            waveform = self.waveform.copy()

        if len(waveform) < 2:
            return

        # Only draw the waveform when a face is detected
        if (
            not self.face_detected
            or self.last_face is None
        ):
            return

        x, y, w, h = self.last_face

        gap = 8
        waveform_height = 55

        waveform_x1 = x
        waveform_x2 = x + w

        waveform_y2 = y - gap
        waveform_y1 = (
            waveform_y2
            - waveform_height
        )

        # Move the waveform if there is not enough space above the face
        if waveform_y1 < 5:

            waveform_y1 = 100

            waveform_y2 = (
                waveform_y1
                + waveform_height
            )

        waveform_x1 = max(
            5,
            waveform_x1
        )

        waveform_x2 = min(
            self.frame_width - 5,
            waveform_x2
        )

        waveform_y1 = max(
            5,
            waveform_y1
        )

        waveform_y2 = min(
            self.frame_height - 5,
            waveform_y2
        )

        if (
            waveform_x2 <= waveform_x1
            or waveform_y2 <= waveform_y1
        ):
            return


        # Find the middle of the waveform area
        center_y = (
            waveform_y1
            + waveform_height // 2
        )

        center_y = min(
            waveform_y2,
            max(
                waveform_y1,
                center_y
            )
        )

        # Draw the middle line
        cv2.line(
            frame,
            (
                waveform_x1,
                center_y
            ),
            (
                waveform_x2,
                center_y
            ),
            (70, 70, 70),
            1,
            cv2.LINE_AA
        )


        # Convert waveform values into screen points
        points = []

        graph_width = (
            waveform_x2
            - waveform_x1
        )

        for i, value in enumerate(
            waveform
        ):

            px = (
                waveform_x1
                + int(
                    i
                    * graph_width
                    / (len(waveform) - 1)
                )
            )

            py = (
                center_y
                - int(value * 18)
            )

            py = max(
                waveform_y1,
                py
            )

            py = min(
                waveform_y2,
                py
            )

            points.append(
                (px, py)
            )

        # Connect all the waveform points
        for i in range(
            1,
            len(points)
        ):

            cv2.line(
                frame,
                points[i - 1],
                points[i],
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        # Add the rPPG label
        cv2.putText(
            frame,
            "rPPG",
            (
                waveform_x1 + 4,
                waveform_y1 + 14
            ),
            self.font,
            0.35,
            (255, 255, 255),
            1,
            self.font_line
        )

    # BPM Calculation
    def calculate_bpm(self):

        # To get enough samples from all three face areas
        with self.lock:

            if (
                len(self.forehead_rgb_values) < 120
                or len(self.left_cheek_rgb_values) < 120
                or len(self.right_cheek_rgb_values) < 120
            ):

                self.bpm_status = (
                    "COLLECTING SIGNAL"
                )

                return

            # Use the most recent samples
            sample_count = min(
                240,
                len(self.forehead_rgb_values),
                len(self.left_cheek_rgb_values),
                len(self.right_cheek_rgb_values),
                len(self.multi_roi_timestamps)
            )

            forehead_rgb = np.array(
                self.forehead_rgb_values[
                    -sample_count:
                ],
                dtype=np.float64
            )

            left_cheek_rgb = np.array(
                self.left_cheek_rgb_values[
                    -sample_count:
                ],
                dtype=np.float64
            )

            right_cheek_rgb = np.array(
                self.right_cheek_rgb_values[
                    -sample_count:
                ],
                dtype=np.float64
            )

            timestamps = np.array(
                self.multi_roi_timestamps[
                    -sample_count:
                ],
                dtype=np.float64
            )

        # Make sure the timestamps are increasing
        if np.any(
            np.diff(timestamps) <= 0
        ):

            self.bpm_status = (
                "WAITING FOR SIGNAL"
            )

            return

        # Find the total recording time
        duration = (
            timestamps[-1]
            - timestamps[0]
        )

        if duration <= 0:

            self.bpm_status = (
                "WAITING FOR SIGNAL"
            )

            return

        # Make the timestamps evenly spaced
        uniform_t = np.linspace(
            timestamps[0],
            timestamps[-1],
            len(timestamps)
        )

        # Function used to make RGB samples evenly spaced
        def resample(rgb):

            return np.column_stack(
                [
                    np.interp(
                        uniform_t,
                        timestamps,
                        rgb[:, c]
                    )
                    for c in range(3)
                ]
            )

        # Resample all three face areas
        forehead_rgb = resample(
            forehead_rgb
        )

        left_cheek_rgb = resample(
            left_cheek_rgb
        )

        right_cheek_rgb = resample(
            right_cheek_rgb
        )

        # Calculate the new recording time
        duration = (
            uniform_t[-1]
            - uniform_t[0]
        )

        if duration <= 0:

            self.bpm_status = (
                "WAITING FOR SIGNAL"
            )

            return

        # Calculate how many samples are taken per second
        sample_fps = (
            len(uniform_t) - 1
        ) / duration

        if sample_fps <= 0:

            self.bpm_status = (
                "WAITING FOR SIGNAL"
            )

            return

        self.sample_fps = sample_fps

        # -----------------------------------------------------
        # THREE ROI rPPG ESTIMATES
        #
        # For each face area:
        # Average the RGB values
        # -Take the green value
        # -Normalize the signal
        # -Remove the overall rise or fall
        # -Apply a 0.7-3.0 Hz filter
        # -Convert the signal using FFT
        # -Find the strongest frequency
        # -Convert it to BPM
        # -----------------------------------------------------

        # Calculate heart rate from the forehead
        forehead_bpm = calculate_rppg(
            forehead_rgb,
            sample_fps
        )

        # Calculate heart rate from the left cheek
        left_cheek_bpm = calculate_rppg(
            left_cheek_rgb,
            sample_fps
        )

        # Calculate heart rate from the right cheek
        right_cheek_bpm = calculate_rppg(
            right_cheek_rgb,
            sample_fps
        )

        # Store the valid BPM values
        bpm_values = []

        for value in (
            forehead_bpm,
            left_cheek_bpm,
            right_cheek_bpm
        ):

            if value is not None:

                value = float(value)

                if np.isfinite(value):

                    bpm_values.append(
                        value
                    )

        # If no valid heart rate was found
        if len(bpm_values) == 0:

            self.bpm_status = (
                "ANALYZING PULSE"
            )

            return

        # Use the middle value of the three BPM results
        combined_bpm = float(
            np.median(bpm_values)
        )

        # Smooth the heart rate so it does not change suddenly
        if self.bpm == 0:

            self.bpm = combined_bpm

        else:

            self.bpm = (
                self.bpm * 0.85
                + combined_bpm * 0.15
            )

        # Heart rate is now ready
        self.bpm_status = (
            "HEART RATE READY"
        )

    def get_data(self):

        # Return the current monitoring information
        with self.lock:

            signal = list(
                self.rgb_values
            )

            samples = len(signal)

            return {
                "bpm": self.bpm,
                "fps": self.fps,
                "face_detected":
                    self.face_detected,
                "roi_detected":
                    self.roi_detected,
                "samples": samples,
                "signal": signal,
                "sample_fps":
                    self.sample_fps,
                "bpm_status":
                    self.bpm_status,
            }
