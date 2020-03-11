const tf = require("@tensorflow/tfjs-node");
const facemesh = require("@tensorflow-models/facemesh");
const fs = require("fs");
const cv = require("opencv4nodejs");

const output = "./point.txt";
// to invest the api for debugging
const getMethods = obj => {
  let properties = new Set();
  let currentObj = obj;
  do {
    Object.getOwnPropertyNames(currentObj).map(item => properties.add(item));
  } while ((currentObj = Object.getPrototypeOf(currentObj)));
  return [...properties.keys()].filter(item => typeof obj[item] === "function");
};

async function main() {
  try {
    fs.unlinkSync(output);
  } catch (err) {}
  const model = await facemesh.load();
  const video = new cv.VideoCapture("./demo.mp4");
  let frame = video.read();
  let frameCount = 0;
  console.log("start processing");
  while (!frame.empty) {
    const imgRow = cv.imencode(".jpg", frame);
    const img = tf.node.decodeImage(imgRow);
    const predictions = await model.estimateFaces(img);
    predictions.map(prediction => {
      const keypointStr = prediction.scaledMesh.reduce(
        (acc, [x, y, z], i) => acc + `Keypoint ${i}: [${x}, ${y}, ${z}] `
      );
      fs.appendFileSync(
        output,
        `frame ${frameCount}, and keypoints is ${keypointStr}`
      );
    });
    frame = video.read();
    frameCount++;
  }
}

main();
