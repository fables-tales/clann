(ns clann.test.core
  (:use [clann.core])
  (:use [clojure.test]))

(deftest sigmoid-of-zero 
  (is 0 (sigmoid 0))) 

(deftest sigmoid-of-1
  (is true (< 0.0001 (Math/abs (- (sigmoid 1) 0.4621117)))))


(deftest test-sum
 (is 7 (sum '(1,2,4))))

(deftest weight-activation
 (is 14 (computeWeightSum '(1,2,4) '(9,2,0.25))))
