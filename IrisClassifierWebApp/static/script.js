const form = document.getElementById("form");
const result = document.getElementById("result");
const errorEl = document.getElementById("error");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorEl.hidden = true;
  const payload = {};
  new FormData(form).forEach((v, k) => (payload[k] = parseFloat(v)));

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Sunucu hatası: " + res.status);
    const data = await res.json();

    document.getElementById("pred").textContent =
      `Iris ${data.prediction} (%${(data.confidence * 100).toFixed(1)})`;
    const bars = document.getElementById("bars");
    bars.innerHTML = "";
    for (const [name, p] of Object.entries(data.probabilities)) {
      const row = document.createElement("div");
      row.className = "row";
      row.innerHTML = `<span>${name}</span><div class="track"><div class="fill"></div></div><span>%${(p * 100).toFixed(1)}</span>`;
      bars.appendChild(row);
      requestAnimationFrame(() => (row.querySelector(".fill").style.width = p * 100 + "%"));
    }
    result.hidden = false;
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.hidden = false;
  }
});
