import cv2
import mediapipe as mp
import speech_recognition as sr
import webbrowser
import threading


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()


video_capture = cv2.VideoCapture(0)


font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
font_color = (0, 0, 255)


recognizer = sr.Recognizer()
microphone = sr.Microphone()


recognize_voice_command = False


def voice_command_thread():
    global recognize_voice_command

    while True:
        if recognize_voice_command:
            with microphone as source:
                print("Listening for voice command...")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5)

            try:
                command = recognizer.recognize_google(audio).lower()
                print(f"Voice command: {command}")

                if "open browser" in command:
                    webbrowser.get("safari").open(
                        "http://www.google.com"
                    )
                elif 'stop' in command:
                    break

            except sr.UnknownValueError:
                print("Could not understand audio.")
            except sr.RequestError as e:
                print(
                    f"Could not request results from Google Speech Recognition service; {e}"
                )
            except Exception as e:
                print(e)

            recognize_voice_command = False

voice_command_thread = threading.Thread(target=voice_command_thread)
voice_command_thread.start()

while True:
    ret, frame = video_capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    result = hands.process(frame)

    if result is not None and result.multi_hand_landmarks:
        num_hands = len(result.multi_hand_landmarks)
        cv2.putText(
            frame,
            f"Hands: {num_hands}",
            (10, 30),
            font,
            font_scale,
            font_color,
            font_thickness,
        )

        for hand_landmarks in result.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(
                    landmark.y * frame.shape[0]
                )
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        recognize_voice_command = True

    num_faces = len(faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red square for face

    cv2.putText(
        frame,
        f"Faces: {num_faces}",
        (1000, 300),
        font,
        font_scale,
        font_color,
        font_thickness,
    )

    cv2.imshow("Face and Hand Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
