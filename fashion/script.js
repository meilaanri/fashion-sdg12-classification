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
  const confidenceEl = document.getElementById("confidence");

  resultEl.innerText = "Loading model, please wait...";
  confidenceEl.innerText = "";

  try {
    const modelPath = window.location.origin + "/tfjs_model/model.json";

    console.log("Loading model from:", modelPath);

    model = await tf.loadLayersModel(modelPath);

    console.log("Model loaded successfully");

    resultEl.innerText = "Model loaded. Please upload an image and click Predict.";
    confidenceEl.innerText = "";
  } catch (error) {
    console.error("Model load error:", error);

    resultEl.innerText = "Model failed to load.";
    confidenceEl.innerText = error.message;
  }
}

loadModel();

document.getElementById("imageUpload").addEventListener("change", function (event) {
  const file = event.target.files[0];
  const preview = document.getElementById("previewImage");
  const resultEl = document.getElementById("result");
  const confidenceEl = document.getElementById("confidence");

  if (file) {
    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";

    resultEl.innerText = "Image uploaded. Click Predict.";
    confidenceEl.innerText = "";
  }
});

async function predictImage() {
  const image = document.getElementById("previewImage");
  const resultEl = document.getElementById("result");
  const confidenceEl = document.getElementById("confidence");

  if (!model) {
    resultEl.innerText = "Model is still loading or failed to load.";
    confidenceEl.innerText = "Please wait or check the model error message.";
    return;
  }

  if (!image.src) {
    resultEl.innerText = "Please upload an image first.";
    confidenceEl.innerText = "";
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

    resultEl.innerText = "Prediction failed.";
    confidenceEl.innerText = error.message;
  }
}
