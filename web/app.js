const ws = new WebSocket("ws://localhost:8000/ws");

let collectInterval = null;
let inferInterval = null;
const sampleRate = 250; // 取樣率
const maxSeconds = 10;  
const maxSamples = sampleRate * maxSeconds; // 2500筆
let collectingTraining = false;
let trainingData = [];
let trainingRounds = 0;
const trainingTotalRounds = 3;
const collectDuration = 10; 
let fullData = [];        // 全部收集到的資料，用於推論與訓練
let chartBuffer = [];     // 抽樣進來的資料，用來畫圖
const chartMaxLength = 250;
let drawCounter = 0;

const chart = new Chart(document.getElementById("chart").getContext("2d"), {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'Real Time Signal',
      borderColor: 'blue',
      borderWidth: 1,
      pointRadius: 0,
      data: [],
    }]
  },
  options: {
    animation: false,
    elements: {
      point: { radius: 0 }
    },
    plugins: {
      legend: { display: false }
    },
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { display: false },
      y: {}
    }
  }
});

// 開始模擬
function startStreaming() {
  chart.data.labels = [];
  chart.data.datasets[0].data = [];
  chart.update();

  if (!window.bleListenerInitialized && window.bleCharacteristic) {
    window.bleCharacteristic.addEventListener('characteristicvaluechanged', (event) => {
      const text = new TextDecoder().decode(event.target.value).trim();
      const value = parseFloat(text);
      if (!isNaN(value)) {
        fullData.push(value);
        drawCounter++;
        if (drawCounter >= 10) {
          chartBuffer.push(value);
          if (chartBuffer.length > chartMaxLength) chartBuffer.shift();
          drawCounter = 0;
        }
      }
      if (fullData.length > maxSamples+500) {
        fullData.shift();
      }
    });
    window.bleCharacteristic.startNotifications(); // 確保啟用通知
    window.bleListenerInitialized = true;
  }
  // 啟動自動推論
  startInferLoop();
}

function renderLoop() {
  if (chartBuffer.length > 0) {
    const nextVal = chartBuffer.shift(); // 取出一筆代表性資料

    chart.data.labels.push('');
    chart.data.datasets[0].data.push(nextVal);

    if (chart.data.labels.length > chartMaxLength) {
      chart.data.labels.shift();
      chart.data.datasets[0].data.shift();
    }

    chart.update();
  }

  requestAnimationFrame(renderLoop);
}

function startInferLoop() {
  if (inferInterval) clearInterval(inferInterval);
  inferInterval = setInterval(() => {
    if (!collectingTraining) {  // 🔥 只有不是訓練中才推論
      collectAndInfer();
    }
  }, 10100); // 每10秒多一點推論一次
}

function collectAndInfer() {
  const fullBuffer = fullData
  if (fullBuffer.length < maxSamples) {
    console.log("Not enough data yet for inference. Current length:", fullBuffer.length);
    return;
  }

  const recentData = fullBuffer.slice(-maxSamples);
  ws.send(JSON.stringify({
    action: "inference",
    data: recentData
  }));
}

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  if (msg.type === "inference_result") {
    const result = Number(msg.result);
    if (result == 1) {
      document.getElementById("warning").innerText = "Warning! Atrial Fibrillation Detected!";
    }else if (result == 2){
      document.getElementById("warning").innerText = "Warning! Premature Atrial Contraction Detected!";
    }else if (result == 3){
      document.getElementById("warning").innerText = "Warning! Premature Ventricular Contraction Detected!";
    }else if (result == 4){
      document.getElementById("warning").innerText = "Slower Heart Rate";
    }else if (result == 5){
      document.getElementById("warning").innerText = "Faster Heart Rate";
    }else {
      document.getElementById("warning").innerText = "";
    }
  }
  if (msg.type === "training_done") {
    document.getElementById("status").innerText = "Training Done! Model Updated!";

    // 5秒後清除「訓練完成」訊息
    setTimeout(() => {
      document.getElementById("status").innerText = "Mode: Inference";
    }, 5000);

    // 訓練完成後恢復推論
    startInferLoop();
  }
};

function startTraining() {
  if (collectingTraining) return;

  document.getElementById("status").innerText = "Mode: Training";
  document.getElementById("warning").innerText = "";
  document.getElementById("collecting-status").innerText = "";

  trainingData = [];
  trainingRounds = 0;

  // 訓練模式開始時，暫停推論
  if (inferInterval) clearInterval(inferInterval);

  let prepareTime = 3; // 3秒準備倒數

  const prepareInterval = setInterval(() => {
    document.getElementById("collecting-status").innerText = `Preparing... ${prepareTime} s Left`;
    prepareTime--;

    if (prepareTime < 0) {
      clearInterval(prepareInterval);
      collectingTraining = true;
      collectTrainingRound();
    }
  }, 1000);
}

function collectTrainingRound() {
  const startTime = Date.now();

  collectInterval = setInterval(() => {
    const elapsed = (Date.now() - startTime) / 1000;
    const remaining = collectDuration - elapsed;

    document.getElementById("collecting-status").innerText =
      `Collecting Data... ${remaining.toFixed(1)} s Left`;

    if (remaining <= 0) {
      clearInterval(collectInterval);

      // 像 inference 模式一樣，從 chart 中擷取最後 2500 筆
      const fullBuffer = fullData
      const recent2500 = fullBuffer.slice(-maxSamples);

      document.getElementById("collecting-status").innerText =
        `No. ${trainingRounds + 1} Batch Collected!`;

      setTimeout(() => {
        if (!collectingTraining) {
          document.getElementById("collecting-status").innerText = "";
        }
      }, 5000);

      trainingData.push(recent2500);
      trainingRounds += 1;

      if (trainingRounds < trainingTotalRounds) {
        setTimeout(collectTrainingRound, 1000);
      } else {
        ws.send(JSON.stringify({
          action: "training",
          data: trainingData,
          label: trainingData.map(() => 0)
        }));

        collectingTraining = false;
        document.getElementById("status").innerText = "Traing Model...";
      }
    }
  }, 100);
}

