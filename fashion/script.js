let model;

const classNames = [
  "T-shirt/top",
  "Trouser",
  "Pullover",
  "Dress",
  "Coat",
  "Sandal",
  "Shirt",
  "Sneaker",
  "Bag",
  "Ankle boot"
];

async function loadModel() {
  const resultEl = document.getElementById("result");

  resultEl.innerText = "Loading model, please wait...";

  try {
    model = await tf.loadLayersModel("./tfjs_model/model.json");
    console.log("Model loaded successfully");
    resultEl.innerText = "Model loaded. Please upload an image and click Predict.";
  } catch (error) {
    console.error("Model load error:", error);
    resultEl.innerText = "Model failed to load. Please check tfjs_model/model.json path.";
  }
}

loadModel();

document.getElementById("imageUpload").addEventListener("change", function (event) {
  const file = event.target.files[0];
  const preview = document.getElementById("previewImage");

  if (file) {
    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";
    document.getElementById("result").innerText = "Image uploaded. Click Predict.";
    document.getElementById("confidence").innerText = "";
  }
});

async function predictImage() {
  const image = document.getElementById("previewImage");
  const resultEl = document.getElementById("result");
  const confidenceEl = document.getElementById("confidence");

  if (!model) {
    resultEl.innerText = "Model is still loading or failed to load.";
    return;
  }

  if (!image.src) {
    resultEl.innerText = "Please upload an image first.";
    return;
  }

  resultEl.innerText = "Predicting...";
  confidenceEl.innerText = "";

  try {
    const tensor = tf.browser.fromPixels(image)
      .resizeNearestNeighbor([28, 28])
      .mean(2)
      .expandDims(2)
      .expandDims(0)
      .toFloat()
      .div(255.0);

    const prediction = model.predict(tensor);
    const predictionData = await prediction.data();

    const maxProbability = Math.max(...predictionData);
    const predictedIndex = predictionData.indexOf(maxProbability);

    resultEl.innerText = "Prediction: " + classNames[predictedIndex];
    confidenceEl.innerText = "Confidence: " + (maxProbability * 100).toFixed(2) + "%";

    tensor.dispose();
    prediction.dispose();
  } catch (error) {
    console.error("Prediction error:", error);
    resultEl.innerText = "Prediction failed. Please check image input or model format.";
  }
}
