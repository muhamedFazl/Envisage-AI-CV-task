import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 1 if you have a second webcam

# Initialize detector with detection confidence and max hands
detector = HandDetector(detectionCon=0.8, maxHands=2)

flag = False
while True:
    # Get image frame
    success, img = cap.read()
    if not success:
        break

    # Find hands in the frame
    hands, img = detector.findHands(img)  # Draw hands landmarks
    # hands = detector.findHands(img, draw=True)  # Draw hands landmarks
    
    # if flag == False:
    #     if hands != []:
    #         print(hands)  # Print the list of detected hands
    #         flag = True

    if hands:
        # For the first hand
        hand1 = hands[0]
        lmList1 = hand1['lmList']           # List of 21 Landmarks (x, y, z)
        bbox1 = hand1['bbox']               # Bounding box info x, y, w, h
        centerPoint1 = hand1['center']      # Center of the hand cx, cy
        handType1 = hand1['type']           # 'Left' or 'Right'

        # Print info about hand1
        # print(f"Hand 1 - Type: {handType1}, Center: {centerPoint1}, BBox: {bbox1}")

        # How many fingers are up?
        fingers1 = detector.fingersUp(hand1)
        # print(f"Hand 1 fingers up: {fingers1}")

        # Calculate distance between index and middle finger tips on hand1
        # length, info, img = detector.findDistance(lmList1[8], lmList1[12], img)
        # print(f"Distance between index and middle finger: {length}")

        # If two hands are detected, process the second one
        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2['lmList']
            centerPoint2 = hand2['center']
            handType2 = hand2['type']

            # print(f"Hand 2 - Type: {handType2}, Center: {centerPoint2}")

            fingers2 = detector.fingersUp(hand2)

            if fingers1 == [0, 1, 0, 0, 0] and fingers2 == [0, 1, 0, 0, 0]:
                # calculate distance between the two index fingers
                hand1_index = lmList1[8]  # Index finger tip of hand 1
                hand2_index = lmList2[8]  # Index finger tip of hand
                length, info, img = detector.findDistance((lmList1[8][0], lmList1[8][1]), (lmList2[8][0], lmList2[8][1]), img)
                print(f"Distance between index fingers: {length}")
            else:
                print("Both hands are not in the correct position (keep only the index fingers up).")

            # Calculate distance between index finger tips of both hands
            # length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
            # print(f"Distance between index finger tips of both hands: {length}")

            # Distance between the center points of both hands
            # length, info, img = detector.findDistance(centerPoint1, centerPoint2, img)
            # print(f"Distance between centers of both hands: {length}")

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
