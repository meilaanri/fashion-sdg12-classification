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
  model = await tf.loadLayersModel("./tfjs_model/model.json");
  console.log("Model loaded successfully");
}

loadModel();

document.getElementById("imageUpload").addEventListener("change", function (event) {
  const file = event.target.files[0];
  const preview = document.getElementById("previewImage");

  if (file) {
    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";
  }
});

async function predictImage() {
  const image = document.getElementById("previewImage");

  if (!model) {
    alert("Model is still loading. Please wait.");
    return;
  }

  if (!image.src) {
    alert("Please upload an image first.");
    return;
  }

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

  document.getElementById("result").innerText =
    "Prediction: " + classNames[predictedIndex];

  document.getElementById("confidence").innerText =
    "Confidence: " + (maxProbability * 100).toFixed(2) + "%";

  tensor.dispose();
  prediction.dispose();
}