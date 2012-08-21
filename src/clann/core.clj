(ns clann.core)

(use 'clojure.contrib.generic.math-functions)

(defn zip [a,b]
  (map vector a b))


(defn zip3 [a,b,c]
  (map vector a b c))

(defn sum [a]
  (reduce + a))

(defn sigmoid [v]
  (/ 1.0 (inc (exp (- v)))))

(defn inverseSigmoid [v]
  (-(log (dec (/ 1.0 v)))))

(defn multiply-pair [pair]
  (* (first pair) (second pair)))

(defn computeWeightSum [inputs, weights]
  (sum (map multiply-pair (zip inputs weights))))


(defn activation [inputs, weights]
  (sigmoid (computeWeightSum inputs weights)))

(defn makeWeights [nWeights]
  (take nWeights (repeatedly #(- 1 (rand 2)))))

(defn makeNetworkLayerWeights [nInputs, nOutputs]
  (take nOutputs (repeatedly #(makeWeights nInputs))))

(defn feedForward [inputs, layerWeights]
  (map #(activation inputs %1) layerWeights))


(defn twoLayerPerceptron [inputs, layer1Weights, layer2Weights]
  (feedForward (feedForward inputs layer1Weights) layer2Weights))


(defn mulEach [a1, a2]
  (map #(* (first %1) (second %1)) (zip a1 a2)))

(def xorExample [[0,0],[0,1],[1,0][1,1]])
(def xorLabels (map #([apply bit-xor %1]) xorExample))

(defn diff [value1, value2]
  (- value1 value2))

(defn ones [n]
  (take n (repeatedly (fn [] 1))))

(defn neuronError [neuronOutput, errorTerm]
  (* neuronOutput
     (- 1 neuronOutput)
     errorTerm))

(defn outputLayerNeuronError [neuronOutput, neuronExpected]
  (neuronError neuronOutput (- neuronExpected neuronOutput)))

(defn allOutputErrors [labels, predictions]
  (map #(outputLayerNeuronError (first %1) (second %1)) (zip predictions labels)))


(defn hiddenLayerError [neuronOutput, neuronWeights, outputLayerErrors]
  (neuronError neuronOutput (sum (mulEach neuronWeights outputLayerErrors))))


(defn allHiddenErrors [hiddenOutputs, hiddenLayerWeights, outputErrors]
  (map #(hiddenLayerError (first %1) (second %1) outputErrors) (zip hiddenOutputs hiddenLayerWeights)))

(defn nextWeights [neuronWeights, error, previousOutput]
  (doall (map #(+ (first %1) (* 0.1 error (second %1))) (zip neuronWeights previousOutput))))


(defn twoLayerEpoch [layerHiddenWeights, layerOutputWeights, sample, label]
  (let [hiddenOutput  (feedForward sample layerHiddenWeights)
        networkOutput (feedForward hiddenOutput layerOutputWeights)
        outputErrors  (allOutputErrors label hiddenOutput)
        hiddenErrors  (allHiddenErrors hiddenOutput layerHiddenWeights outputErrors)]
    [(doall (map #(nextWeights (first %1) (second %1) sample) (zip layerHiddenWeights hiddenErrors))),
     (doall (map #(nextWeights (first %1) (second %1) hiddenOutput) (zip layerOutputWeights outputErrors)))]))


(defn trainTwoLayerPerceptron [nEpochs, weights, sample, label]
  (loop [cnt nEpochs weights weights]
    (if
      (zero? cnt)
      weights
      (recur (dec cnt) (twoLayerEpoch (first weights) (second weights) sample label)))))

(defn doEpochs [n, layerWeights, sample, label]
  (loop [cnt n layerWeights layerWeights]
    (if
      (zero? cnt)
      layerWeights
      (let [prediction (feedForward sample layerWeights)
           error (allOutputErrors label prediction)]
        (recur (dec cnt)
               (doall (map #(nextWeights (first %1) (second %1) sample) (zip layerWeights error))))))))
