
package com.example.android.camerax.tflite

import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage

/**
 * Helper class used to communicate between our app and the TF object detection model
 */
class ObjectDetectionHelper(private val tflite: Interpreter, private val labels: List<String>) {

    /** Abstraction object that wraps a prediction output in an easy to parse way */
    data class ObjectPrediction(val label: String, val score: Float)
    // RectF : top,left,right,bottom 의 정보를 담고 있는 직사각형 클래스
    private val labelIndices =  arrayOf(FloatArray(OBJECT_COUNT)) // 라벨 결정자 들어갈 변수 배열로 초기화
    private val scores =  arrayOf(FloatArray(OBJECT_COUNT)) // 점수가 들어갈 변수 배열로 초기화

    private val outputBuffer = mapOf( //출력버퍼 초기화. 0은 위치, 1은 라벨, 2는 점수, 3은 ??)
        1 to labelIndices,
        2 to scores,
        3 to FloatArray(1)
    )

    val predictions get() = (0 until OBJECT_COUNT).map {//10개(obj count) 까지 입력가능
        ObjectPrediction(
            // SSD Mobilenet V1 Model assumes class 0 is background class
            // in label file and class labels start from 1 to number_of_classes + 1,
            // while outputClasses correspond to class index from 0 to number_of_classes
            label = labels[1+labelIndices[0][it].toInt()],
            //labels.txt 에서 predict 된 index 값의 라벨을 가져온다.
            // 점수는 0~1 사이의 single value.
            score = scores[0][it]
        )
    }

    fun predict(image: TensorImage): List<ObjectPrediction> {
       tflite.runForMultipleInputsOutputs(arrayOf(image.buffer), outputBuffer)
        //dectect된 여러 객체의 정보를 모델에 push
        //안전,위험의 경우 그냥 tflite.process(arrayOf(image.buffer)를 쓰면 되지 않을까?
        return predictions //결과를 리스트형태로 출력.
    }

    companion object { //상수 처리
        const val OBJECT_COUNT = 10 //const: 런타임이 아닌, 컴파일 타임에 값 초기화.//최대 오브젝트 갯수설정
    }
}