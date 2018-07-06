var X = [[0,0],[0,1],[1,0],[1,1]]
var Y = [[0],[1],[1],[0]]

const xs = tf.tensor2d(X)
const ys = tf.tensor2d(Y)

const deep_model = tf.sequential()
deep_model.add(tf.layers.dense({units:20, inputShape:2, activation: 'tanh'}))
deep_model.add(tf.layers.dense({units:1, activation: 'sigmoid'}))
deep_model.compile({optimizer: 'sgd', loss: 'binaryCrossentropy', lr:0.1})

async function train () {
  console.log('here');

  await deep_model.fit(xs, ys, {
    batchSize: 4,
    epochs: 1
  })
}


async function mydraw() {
	// tf.tidy(() => {
		weights = deep_model.getWeights()[0].clone()
		max = weights.max()
		min = weights.min()

		final_weights = weights.sub(min).div(max.sub(min))

		tf.toPixels(final_weights, canvas)
	// })

	// tf.toPixels(tf.randomUniform([100, 100]), canvas)
}

var canvas = document.getElementById('weights')

var iteration = 0

function setup () {
  setInterval(train, 5)
  setInterval(mydraw, 5)
  // createCanvas(0, 400)
  frameRate(10)

  noLoop()
}

function draw () {
  // background(51)
  //
  // fill(255)
  // stroke(251)
  // textSize(25)
  //
  // text('XOR', 50, 50)
  //
  // stroke(51)

  // tf.tidy(() => {
	// 	train()
	//   	// deep_model.getWeights()[0].print()
	//   	tf.toPixels(deep_model.getWeights()[0], canvas).then(() => {
	// 		console.log('drawn');
	//   	}).catch(() => {
  //   		console.log('not drawn');
  // 		})
  // });

  // text('Iteration : ' + iteration++, 50, 100)
  // text('No of Tensors : ' + tf.memory().numTensors, 50, 150)
  //
  // textSize(20)
  //
  // tf.tidy(() => {
  //   const yy = deep_model.predict(xs)
  //   text('Prediction : \n\n\t' + yy, 50, 200)
  //   yy.dispose()
  // });

  // Check Memory Leak
  console.log(tf.memory().numTensors);
}
