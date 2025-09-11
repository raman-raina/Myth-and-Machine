async function sendData() {
  const fileInput = document.getElementById("imageInput").files[0];

  if (!fileInput) {
    alert("Please upload an image.");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput);

  const backendURL = "https://rune-classifier.onrender.com/predict";

  try {
    const response = await fetch(backendURL, {
      method: "POST",
      body: formData
    });

    const data = await response.json();
    document.getElementById("output").innerText = data.result;

  } catch (error) {
    document.getElementById("output").innerText = "Error: " + error.message;
  }

}
