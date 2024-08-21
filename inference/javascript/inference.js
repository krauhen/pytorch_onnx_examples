const ort = require('onnxruntime-web');
const Jimp = require('jimp');

async function runInference() {
    // Load the ONNX model
    const session = await ort.InferenceSession.create('../../models/model.with_pre_post_processing.onnx');

    // Read and preprocess the image
    const image = await Jimp.read('../../imgs/2.jpg');
    image.grayscale(); // Convert to grayscale

    // Convert the image to a tensor
    const width = image.bitmap.width;
    const height = image.bitmap.height;
    const imageData = new Uint8Array(width * height * 1); // 1 channel for grayscale

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const pixel = image.getPixelColor(x, y);
            const grayValue = Jimp.intToRGBA(pixel).r; // Get the grayscale value (0-255)
            imageData[y * width + x] = grayValue;
        }
    }

    // Create the input tensor
    const tensor = new ort.Tensor('uint8', imageData, [height, width, 1]);

    // Get model input name
    const inputName = session.inputNames[0];

    // Run the model
    const feeds = { [inputName]: tensor };
    const results = await session.run(feeds);

    // Print the result
    console.log(results);
}

runInference().catch(err => {
    console.error(err);
});