package com.example.voiceassistant

import android.content.Context
import android.util.Log
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp

class NLUService(private val context: Context) {

    private val TAG = "NLUService"
    private lateinit var interpreter: Interpreter
    private lateinit var vocab: Map<String, Int>
    private lateinit var labelMap: Map<Int, String>
    private val maxLength = 32

    // Special token IDs (MobileBERT)
    private val CLS_TOKEN_ID = 101
    private val SEP_TOKEN_ID = 102
    private val PAD_TOKEN_ID = 0
    private val UNK_TOKEN_ID = 100

    init {
        try {
            Log.d(TAG, "=".repeat(60))
            Log.d(TAG, "Initializing NLU Service")
            Log.d(TAG, "=".repeat(60))

            loadModel()
            loadVocabulary()
            loadLabelMap()

            // CRITICAL: Log input tensor order
            logTensorDetails()

            // Run test to verify model works
            runInitialTests()

            Log.d(TAG, "✅ NLU Service initialized successfully")
            Log.d(TAG, "=".repeat(60))
        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to initialize NLU Service", e)
            throw e
        }
    }

    private fun loadModel() {
        Log.d(TAG, "Loading TFLite model...")
        val modelPath = "models/nlu/hotel_mobilebert.tflite"
        val modelFile = loadModelFile(modelPath)

        interpreter = Interpreter(modelFile)

        val inputCount = interpreter.inputTensorCount
        val outputCount = interpreter.outputTensorCount

        Log.d(TAG, "✅ Model loaded:")
        Log.d(TAG, "   Input tensors: $inputCount")
        Log.d(TAG, "   Output tensors: $outputCount")
    }

    private fun loadVocabulary() {
        Log.d(TAG, "Loading vocabulary...")
        val vocabPath = "models/nlu/vocab.json"
        val vocabString = context.assets.open(vocabPath).bufferedReader().use { it.readText() }
        val vocabJson = JSONObject(vocabString)

        vocab = mutableMapOf()
        val keys = vocabJson.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            vocab = vocab + (key to vocabJson.getInt(key))
        }

        Log.d(TAG, "✅ Vocabulary loaded: ${vocab.size} tokens")

        // Verify special tokens
        Log.d(TAG, "   [CLS]: ${vocab["[CLS]"]}")
        Log.d(TAG, "   [SEP]: ${vocab["[SEP]"]}")
        Log.d(TAG, "   [PAD]: ${vocab["[PAD]"]}")
        Log.d(TAG, "   [UNK]: ${vocab["[UNK]"]}")
    }

    private fun loadLabelMap() {
        Log.d(TAG, "Loading label map...")
        val labelMapPath = "models/nlu/label_map.json"
        val labelMapString = context.assets.open(labelMapPath).bufferedReader().use { it.readText() }
        val labelMapJson = JSONObject(labelMapString)

        labelMap = mutableMapOf()
        val keys = labelMapJson.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            labelMap = labelMap + (key.toInt() to labelMapJson.getString(key))
        }

        Log.d(TAG, "✅ Label map loaded: ${labelMap.size} intents")
        labelMap.toSortedMap().forEach { (idx, intent) ->
            Log.d(TAG, "   $idx: $intent")
        }
    }

    private fun logTensorDetails() {
        Log.d(TAG, "")
        Log.d(TAG, "=".repeat(60))
        Log.d(TAG, "TENSOR DETAILS (CRITICAL FOR DEBUGGING)")
        Log.d(TAG, "=".repeat(60))

        // Log input tensors
        for (i in 0 until interpreter.inputTensorCount) {
            val tensor = interpreter.getInputTensor(i)
            Log.d(TAG, "Input $i:")
            Log.d(TAG, "   Name: ${tensor.name()}")
            Log.d(TAG, "   Shape: ${tensor.shape().contentToString()}")
            Log.d(TAG, "   Type: ${tensor.dataType()}")
        }

        // Log output tensors
        for (i in 0 until interpreter.outputTensorCount) {
            val tensor = interpreter.getOutputTensor(i)
            Log.d(TAG, "Output $i:")
            Log.d(TAG, "   Name: ${tensor.name()}")
            Log.d(TAG, "   Shape: ${tensor.shape().contentToString()}")
            Log.d(TAG, "   Type: ${tensor.dataType()}")
        }

        Log.d(TAG, "")
        Log.d(TAG, "EXPECTED FROM PYTHON:")
        Log.d(TAG, "   Input 0: serving_default_attention_mask:0")
        Log.d(TAG, "   Input 1: serving_default_input_ids:0")
        Log.d(TAG, "=".repeat(60))
    }

    private fun runInitialTests() {
        Log.d(TAG, "")
        Log.d(TAG, "Running initial tests...")
        Log.d(TAG, "-".repeat(60))

        // Test with the exact failing examples from your dashboard
        val testCases = listOf(
            "may i request room cleaning service" to "room_cleaning",
            "i need a coffee" to "food_order",
            "housekeeping please" to "room_cleaning"
        )

        var passCount = 0
        testCases.forEach { (text, expected) ->
            val (predicted, confidence) = classifyIntent(text, debug = true)
            val status = if (predicted == expected) {
                passCount++
                "✅ PASS"
            } else {
                "❌ FAIL"
            }
            Log.d(TAG, "$status - '$text'")
            Log.d(TAG, "   Expected: $expected")
            Log.d(TAG, "   Got: $predicted (${"%.1f".format(confidence * 100)}%)")
            Log.d(TAG, "")
        }

        Log.d(TAG, "Initial test results: $passCount/${testCases.size} passed")
        Log.d(TAG, "-".repeat(60))

        if (passCount != testCases.size) {
            Log.e(TAG, "⚠️ WARNING: Initial tests failed!")
            Log.e(TAG, "   Check tensor order in classifyIntent()")
        }
    }

    /**
     * Classify guest intent from text
     * @param text The user's input text
     * @param debug Enable detailed logging
     * @return Pair of (intent, confidence)
     */
    fun classifyIntent(text: String, debug: Boolean = false): Pair<String, Float> {
        if (debug) {
            Log.d(TAG, "")
            Log.d(TAG, "=".repeat(60))
            Log.d(TAG, "CLASSIFICATION DEBUG")
            Log.d(TAG, "=".repeat(60))
        }

        try {
            // Step 1: Preprocess text - ALWAYS lowercase
            val normalizedText = preprocessText(text)

            if (debug) {
                Log.d(TAG, "Raw input: '$text'")
                Log.d(TAG, "Normalized: '$normalizedText'")
            }

            // Step 2: Tokenize
            val tokens = tokenize(normalizedText)

            if (debug) {
                Log.d(TAG, "Tokens: ${tokens.take(10)}${if (tokens.size > 10) "..." else ""}")
                Log.d(TAG, "Token count: ${tokens.size}")
            }

            // Step 3: Create input tensors
            val (inputIds, attentionMask) = createInputTensors(tokens)

            if (debug) {
                Log.d(TAG, "Input IDs: ${inputIds.take(10).toList()}")
                Log.d(TAG, "Attention: ${attentionMask.take(10).toList()}")
                Log.d(TAG, "")
                Log.d(TAG, "EXPECTED (from Python):")
                Log.d(TAG, "   InputIds: [101, 2089, 1045, 5227, 2282, 9344, 2326, 102, 0, 0]")
                Log.d(TAG, "   Attention: [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]")
                Log.d(TAG, "")
            }

            // Step 4: Run inference
            val logits = runInference(inputIds, attentionMask)

            if (debug) {
                Log.d(TAG, "Raw logits (first 5): ${logits.take(5).map { "%.2f".format(it) }}")
            }

            // Step 5: Apply softmax
            val probabilities = softmax(logits)

            // Step 6: Get prediction
            val predictedIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
            val confidence = probabilities[predictedIndex]
            val intent = labelMap[predictedIndex] ?: "unknown"

            if (debug) {
                Log.d(TAG, "")
                Log.d(TAG, "Top 5 predictions:")
                val top5 = probabilities.indices
                    .sortedByDescending { probabilities[it] }
                    .take(5)

                top5.forEachIndexed { rank, idx ->
                    val intentName = labelMap[idx] ?: "unknown"
                    val prob = probabilities[idx]
                    Log.d(TAG, "   ${rank + 1}. $intentName: ${"%.2f".format(prob * 100)}%")
                }

                Log.d(TAG, "")
                Log.d(TAG, "Final prediction: $intent (${"%.2f".format(confidence * 100)}%)")
                Log.d(TAG, "=".repeat(60))
            }

            return Pair(intent, confidence)

        } catch (e: Exception) {
            Log.e(TAG, "Error during classification", e)
            return Pair("error", 0.0f)
        }
    }

    /**
     * Preprocess text: lowercase and normalize whitespace
     */
    private fun preprocessText(text: String): String {
        return text.lowercase()
            .trim()
            .replace(Regex("\\s+"), " ")
    }

    /**
     * Simple word-based tokenization
     * For production, use proper WordPiece tokenization
     */
    private fun tokenize(text: String): List<Int> {
        val tokens = mutableListOf<Int>()

        text.split(" ").forEach { word ->
            // Try to get token ID, fallback to [UNK]
            val tokenId = vocab[word] ?: vocab["[UNK]"] ?: UNK_TOKEN_ID
            tokens.add(tokenId)
        }

        return tokens
    }

    /**
     * Create input tensors with special tokens
     * Returns: Pair of (inputIds, attentionMask)
     */
    private fun createInputTensors(tokens: List<Int>): Pair<IntArray, IntArray> {
        val inputIds = IntArray(maxLength) { PAD_TOKEN_ID }
        val attentionMask = IntArray(maxLength) { 0 }

        // Add [CLS] token at start
        inputIds[0] = CLS_TOKEN_ID
        attentionMask[0] = 1

        // Add tokens (leave room for [SEP])
        val maxTokens = minOf(tokens.size, maxLength - 2)
        tokens.take(maxTokens).forEachIndexed { index, tokenId ->
            inputIds[index + 1] = tokenId
            attentionMask[index + 1] = 1
        }

        // Add [SEP] token at end
        val sepPosition = minOf(tokens.size + 1, maxLength - 1)
        inputIds[sepPosition] = SEP_TOKEN_ID
        attentionMask[sepPosition] = 1

        return Pair(inputIds, attentionMask)
    }

    /**
     * Run TFLite inference
     * PERMANENT FIX: Using array for inputs, map for outputs
     */
    private fun runInference(inputIds: IntArray, attentionMask: IntArray): FloatArray {
        try {
            // Prepare inputs as 2D arrays with shape [1, 32]
            val attentionMaskInput = Array(1) { attentionMask }
            val inputIdsInput = Array(1) { inputIds }

            // Prepare output
            val outputArray = Array(1) { FloatArray(labelMap.size) }

            // Create inputs array for runForMultipleInputsOutputs
            val inputsArray = arrayOf<Any>(attentionMaskInput, inputIdsInput)

            // Create outputs map
            val outputsMap = mutableMapOf<Int, Any>()
            outputsMap[0] = outputArray

            // Run inference
            interpreter.runForMultipleInputsOutputs(inputsArray, outputsMap)

            return outputArray[0]

        } catch (e: Exception) {
            Log.e(TAG, "Inference error: ${e.message}", e)
            throw e
        }
    }

    /**
     * Apply softmax to convert logits to probabilities
     */
    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val expValues = logits.map { exp((it - maxLogit).toDouble()).toFloat() }
        val sumExp = expValues.sum()
        return expValues.map { it / sumExp }.toFloatArray()
    }

    /**
     * Load TFLite model file from assets
     */
    private fun loadModelFile(assetPath: String): MappedByteBuffer {
        context.assets.openFd(assetPath).use { fileDescriptor ->
            FileInputStream(fileDescriptor.fileDescriptor).use { inputStream ->
                val fileChannel = inputStream.channel
                val startOffset = fileDescriptor.startOffset
                val declaredLength = fileDescriptor.declaredLength
                return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
            }
        }
    }

    /**
     * Public method to enable debug mode for specific classification
     */
    fun classifyWithDebug(text: String): Pair<String, Float> {
        return classifyIntent(text, debug = true)
    }

    /**
     * Get model info for debugging
     */
    fun getModelInfo(): String {
        return """
            NLU Model Info:
            - Intents: ${labelMap.size}
            - Vocabulary: ${vocab.size} tokens
            - Max length: $maxLength
            - Model inputs: ${interpreter.inputTensorCount}
            - Model outputs: ${interpreter.outputTensorCount}
        """.trimIndent()
    }

    /**
     * Test specific phrases (for debugging)
     */
    fun runDebugTests() {
        Log.d(TAG, "")
        Log.d(TAG, "=".repeat(60))
        Log.d(TAG, "RUNNING DEBUG TESTS")
        Log.d(TAG, "=".repeat(60))

        val testPhrases = listOf(
            "may i request room cleaning service",
            "i need a coffee",
            "housekeeping please",
            "bring me pillows",
            "emergency help",
            "turn off lights"
        )

        testPhrases.forEach { phrase ->
            Log.d(TAG, "")
            classifyIntent(phrase, debug = true)
        }
    }

    fun close() {
        interpreter.close()
        Log.d(TAG, "NLU Service closed")
    }
}

/**
 * Usage Example:
 *
 * // Initialize
 * val nluService = NLUService(context)
 *
 * // Run initial debug tests
 * nluService.runDebugTests()
 *
 * // Normal classification
 * val (intent, confidence) = nluService.classifyIntent("clean my room")
 *
 * // Debug specific classification
 * val result = nluService.classifyWithDebug("may i request room cleaning service")
 *
 * // Get model info
 * Log.d("App", nluService.getModelInfo())
 */