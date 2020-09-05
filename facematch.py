import face_recognition

image_of_June= face_recognition.load_image_file('June.JPG')
sample_face_encoding = face_recognition.face_encodings(image_of_June)[0]

unknown_image = face_recognition.load_image_file('June1.JPG')
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]


known_encoding=[sample_face_encoding]



results = face_recognition.compare_faces(
    [sample_face_encoding],unknown_face_encoding)

#get the percentage
face_distances = face_recognition.face_distance(known_encoding,unknown_face_encoding )

print(results[0])
if results[0]:
    print('This is June')
else:
    print("this is not June")


for i, face_distance in enumerate(face_distances):
    print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
    print("Are they identical? {}".format(face_distance < 0.6))
    print()