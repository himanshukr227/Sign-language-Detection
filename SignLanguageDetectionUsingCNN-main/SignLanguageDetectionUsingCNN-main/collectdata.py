import os
import cv2

directory = 'SignImage48x48'  # no trailing slash, easier to join
print("Working dir:", os.getcwd())

# create base folder and subfolders A..Z + blank
if not os.path.exists(directory):
    os.mkdir(directory)
if not os.path.exists(os.path.join(directory, 'blank')):
    os.mkdir(os.path.join(directory, 'blank'))

for i in range(65, 91):  # A..Z
    letter = chr(i)
    path = os.path.join(directory, letter)
    if not os.path.exists(path):
        os.mkdir(path)

# open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera. Try different index (0,1,2) or check camera permissions.")
    raise SystemExit

print("Camera opened. Press keys a..z to save letters, '.' to save blank, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # compute counts for display and filename uniqueness
    count = { letter.lower(): len(os.listdir(os.path.join(directory, letter))) for letter in [chr(i) for i in range(65,91)] }
    count['blank'] = len(os.listdir(os.path.join(directory, 'blank')))

    # show rectangle and two windows
    cv2.rectangle(frame, (0, 40), (300, 300), (255, 255, 255), 2)
    cv2.putText(frame, "Press a..z to save | . = blank | q = quit", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    cv2.imshow("data", frame)

    roi = frame[40:300, 0:300]  # ROI
    cv2.imshow("ROI", roi)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))

    key = cv2.waitKey(10) & 0xFF

    if key == ord('q'):  # quit
        break

    # save if a..z
    if ord('a') <= key <= ord('z'):
        letter = chr(key).upper()
        folder = os.path.join(directory, letter)
        filename = f"{len(os.listdir(folder))}.jpg"  # gets latest count
        save_path = os.path.join(folder, filename)
        cv2.imwrite(save_path, resized)
        print("Saved:", save_path)

    if key == ord('.'):  # blank
        folder = os.path.join(directory, 'blank')
        filename = f"{len(os.listdir(folder))}.jpg"
        save_path = os.path.join(folder, filename)
        cv2.imwrite(save_path, resized)
        print("Saved blank:", save_path)

# cleanup
cap.release()
cv2.destroyAllWindows()
