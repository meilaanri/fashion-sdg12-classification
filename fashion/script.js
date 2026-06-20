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

function patchKeras3ModelConfig(obj) {
  if (Array.isArray(obj)) {
    obj.forEach(patchKeras3ModelConfig);
    return;
  }

  if (obj && typeof obj === "object") {
    if (obj.batch_shape && !obj.batch_input_shape) {
      obj.batch_input_shape = obj.batch_shape;
      delete obj.batch_shape;
    }

    if (
      obj.dtype &&
      typeof obj.dtype === "object" &&
      obj.dtype.class_name === "DTypePolicy" &&
      obj.dtype.config &&
      obj.dtype.config.name
    ) {
      obj.dtype = obj.dtype.config.name;
    }

    Object.keys(obj).forEach((key) => {
      patchKeras3ModelConfig(obj[key]);
    });
  }
}

async function loadModel() {
  const resultEl = document.getElementById("result");
  const confidenceEl = document.getElementById("confidence");

  resultEl.innerText = "Loading model, please wait...";
  confidenceEl.innerText = "";

  try {
    const basePath = window.location.origin + "/tfjs_model/";
    const modelJsonUrl = basePath + "model.json";

    console.log("Loading model json from:", modelJsonUrl);

    const response = await fetch(modelJsonUrl);

    if (!response.ok) {
      throw new Error("model.json not found. Status: " + response.status);
    }

    const modelJson = await response.json();

    patchKeras3ModelConfig(modelJson.modelTopology);

    const weightSpecs = [];
    const weightBuffers = [];

    for (const group of modelJson.weightsManifest) {
      weightSpecs.push(...group.weights);

      for (const path of group.paths) {
        const weightUrl = basePath + path;
        console.log("Loading weight file from:", weightUrl);

        const weightResponse = await fetch(weightUrl);

        if (!weightResponse.ok) {
          throw new Error(path + " not found. Status: " + weightResponse.status);
        }

        const buffer = await weightResponse.arrayBuffer();
        weightBuffers.push(new Uint8Array(buffer));
      }
    }

    let totalLength = 0;
    weightBuffers.forEach((buffer) => {
      totalLength += buffer.length;
    });

    const combinedWeights = new Uint8Array(totalLength);
    let offset = 0;

    weightBuffers.forEach((buffer) => {
      combinedWeights.set(buffer, offset);
      offset += buffer.length;
    });

    const modelArtifacts = {
      modelTopology: modelJson.modelTopology,
      weightSpecs: weightSpecs,
      weightData: combinedWeights.buffer
    };

    model = await tf.loadLayersModel(tf.io.fromMemory(modelArtifacts));

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
    confidenceEl.innerText = "Please wait until the model is loaded.";
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
