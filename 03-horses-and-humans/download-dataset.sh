
# training images
mkdir -p ./.dataset/training
wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip \
    -O ./horse-or-human.zip
unzip ./horse-or-human.zip -d ./.dataset/training
rm -rf ./horse-or-human.zip

# validation images
mkdir -p ./.dataset/validation
wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip \
    -O ./validation-horse-or-human.zip
unzip ./validation-horse-or-human.zip -d ./.dataset/validation
rm -rf ./validation-horse-or-human.zip
