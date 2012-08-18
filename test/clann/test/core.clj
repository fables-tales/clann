(ns clann.test.core
  (:use [clann.core])
  (:use [clojure.test]))

(deftest sigmoid-inverse-sigmoid
         (is (> 0.000001 (Math/abs (- 0.5 (inverseSigmoid (sigmoid 0.5))))))
         (is (> 0.000001 (let [r (rand)] (- r (inverseSigmoid (sigmoid r)))))))

(deftest test-sum
 (is (= 7 (sum '(1,2,4)))))

(deftest weight-activation
 (is (= 14 (computeWeightSum '(1,2,4) '(9,2,0.25)))))

(deftest activation-test
         (is (= 1 (activation '(100) '(100))))
         (is (= 0.5 (activation '(0) '(0))))
         (is (< 0.5 (activation '(1) '(2))))
         (is (< 0.5 (activation '(1) '(0.3)))))

(deftest make-weights-test
         (is (= 10 (count (makeWeights 10)))))

(deftest make-network-layer-weights-test
         (is (= 3 (count (makeNetworkLayerWeights 1 3))))
         (is (= 1 (count (first  (makeNetworkLayerWeights 1 3)  ))))
         (is (= 1 (count (second (makeNetworkLayerWeights 1 3)  ))))
         (is (= 1 (count (nth    (makeNetworkLayerWeights 1 3) 2)))))

(deftest feed-forward-test
         (is (= [1,1] (feedForward [1,2] [[100,0],[0,50]])))
         (is (= [0.5, 0.5] (feedForward [1,2] [[0,0],[0,0]]))))


(deftest propagate-simple-test
         (is (> 0.99 (first (feedForward sample (doEpochs 1000))))))
