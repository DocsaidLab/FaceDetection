set -e

docker build -f Dockerfile \
    --build-arg work_folder=$PWD/.. \
    -t face_detection_dev .