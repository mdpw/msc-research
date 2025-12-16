package com.example.voiceassistant

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.json.JSONObject
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.nio.ByteBuffer
import java.nio.ByteOrder

// Data class for Intent results
data class Intent(
    val name: String,
    val confidence: Float,
    val entities: Map<String, Any> = emptyMap()
)

class NLUService(private val context: Context) {

    private lateinit var interpreter: Interpreter
    private lateinit var vocab: Map<String, Int>
    private lateinit var labelMap: Map<Int, String>
    private val TAG = "NLUService"

    private val MAX_SEQ_LENGTH = 32

    private val CLS_TOKEN = "[CLS]"
    private val SEP_TOKEN = "[SEP]"
    private val UNK_TOKEN = "[UNK]"
    private val PAD_TOKEN = "[PAD]"

    fun initialize() {
        try {
            Log.d(TAG, "🔧 Initializing NLU Service...")

            // Load TFLite model
            val modelFile = loadModelFile("models/nlu/hotel_mobilebert.tflite")

            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }
            interpreter = Interpreter(modelFile, options)

            Log.d(TAG, "✅ TFLite model loaded")

            // Load vocabulary
            val vocabJson = context.assets.open("models/nlu/vocab.json")
                .bufferedReader().use { it.readText() }
            vocab = JSONObject(vocabJson).let { json ->
                json.keys().asSequence().associateWith { json.getInt(it) }
            }

            Log.d(TAG, "✅ Vocabulary loaded: ${vocab.size} tokens")

            // Load label mapping
            val labelJson = context.assets.open("models/nlu/label_map.json")
                .bufferedReader().use { it.readText() }
            labelMap = JSONObject(labelJson).let { json ->
                json.keys().asSequence().associate { it.toInt() to json.getString(it) }
            }

            Log.d(TAG, "✅ NLU initialized - ${labelMap.size} intents")

        } catch (e: Exception) {
            Log.e(TAG, "❌ NLU init failed: ${e.message}", e)
            throw e
        }
    }

    private fun loadModelFile(modelPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun classifyIntent(text: String): Intent {
        try {
            Log.d(TAG, "🔍 Classifying: '$text'")

            // Tokenize input
            val tokens = tokenize(text.lowercase())
            Log.d(TAG, "🔤 Generated ${tokens.size} tokens")

            // Prepare input_ids as ByteBuffer
            val inputIdsBuffer = ByteBuffer.allocateDirect(MAX_SEQ_LENGTH * 4).apply {
                order(ByteOrder.nativeOrder())
            }

            tokens.forEachIndexed { index, token ->
                if (index < MAX_SEQ_LENGTH) {
                    inputIdsBuffer.putInt(token)
                }
            }
            // Pad remaining positions
            for (i in tokens.size until MAX_SEQ_LENGTH) {
                inputIdsBuffer.putInt(0)
            }
            inputIdsBuffer.rewind()

            // Prepare attention_mask as ByteBuffer
            val attentionMaskBuffer = ByteBuffer.allocateDirect(MAX_SEQ_LENGTH * 4).apply {
                order(ByteOrder.nativeOrder())
            }

            for (i in 0 until MAX_SEQ_LENGTH) {
                if (i < tokens.size) {
                    attentionMaskBuffer.putInt(1)
                } else {
                    attentionMaskBuffer.putInt(0)
                }
            }
            attentionMaskBuffer.rewind()

            Log.d(TAG, "📥 Prepared inputs: ${tokens.size} active tokens")

            // Prepare output buffer
            val outputBuffer = ByteBuffer.allocateDirect(labelMap.size * 4).apply {
                order(ByteOrder.nativeOrder())
            }

            // Create input/output arrays for TFLite
            val inputs = arrayOf<Any>(inputIdsBuffer, attentionMaskBuffer)
            val outputs = mutableMapOf<Int, Any>()
            outputs[0] = outputBuffer

            Log.d(TAG, "🤖 Running inference...")

            // Run inference
            interpreter.runForMultipleInputsOutputs(inputs, outputs)

            Log.d(TAG, "✅ Inference complete")

            // Read output scores from buffer
            outputBuffer.rewind()
            val scores = FloatArray(labelMap.size)
            for (i in scores.indices) {
                scores[i] = outputBuffer.getFloat()
            }

            // Calculate probabilities
            val probabilities = softmax(scores)

            // Get top prediction
            val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
            val confidence = probabilities[maxIndex]

            val intentName = labelMap[maxIndex] ?: "unknown"

            // Extract entities
            val entities = extractEntities(text, intentName)

            Log.d(TAG, "✅ Intent: $intentName (confidence: ${(confidence * 100).toInt()}%)")

            return Intent(
                name = intentName,
                confidence = confidence,
                entities = entities
            )

        } catch (e: Exception) {
            Log.e(TAG, "❌ Classification failed: ${e.message}", e)
            e.printStackTrace()
            return Intent(name = "unknown", confidence = 0f)
        }
    }

    private fun tokenize(text: String): List<Int> {
        val tokens = mutableListOf<Int>()

        // Add [CLS] token
        tokens.add(vocab[CLS_TOKEN] ?: 101)

        // Tokenize words
        val words = text.split(Regex("\\s+"))
        for (word in words) {
            val tokenId = vocab[word] ?: vocab[UNK_TOKEN] ?: 100
            tokens.add(tokenId)

            if (tokens.size >= MAX_SEQ_LENGTH - 1) break
        }

        // Add [SEP] token
        tokens.add(vocab[SEP_TOKEN] ?: 102)

        return tokens
    }

    private fun softmax(scores: FloatArray): FloatArray {
        val maxScore = scores.maxOrNull() ?: 0f
        val expScores = scores.map { Math.exp((it - maxScore).toDouble()).toFloat() }
        val sumExp = expScores.sum()
        return expScores.map { it / sumExp }.toFloatArray()
    }

    private fun extractEntities(text: String, intent: String): Map<String, Any> {
        val entities = mutableMapOf<String, Any>()
        val lowerText = text.lowercase()

        when (intent) {
            "towel_request", "pillow_request", "blanket_request" -> {
                // Extract numeric quantity
                val quantityRegex = """(\d+)""".toRegex()
                quantityRegex.find(lowerText)?.let {
                    entities["quantity"] = it.value.toInt()
                    return entities
                }

                // Extract number words
                val numberWords = mapOf(
                    "one" to 1, "a" to 1, "single" to 1,
                    "two" to 2, "couple" to 2,
                    "three" to 3, "four" to 4, "five" to 5,
                    "six" to 6, "seven" to 7, "eight" to 8,
                    "nine" to 9, "ten" to 10,
                    "few" to 2, "several" to 3, "many" to 5
                )

                for ((word, number) in numberWords) {
                    if (lowerText.contains("\\b$word\\b".toRegex())) {
                        entities["quantity"] = number
                        break
                    }
                }
            }

            "food_order" -> {
                // Extract meal type
                when {
                    lowerText.contains("breakfast") -> entities["meal"] = "breakfast"
                    lowerText.contains("lunch") -> entities["meal"] = "lunch"
                    lowerText.contains("dinner") -> entities["meal"] = "dinner"
                    lowerText.contains("snack") -> entities["meal"] = "snack"
                }

                // Extract food items
                val foodItems = listOf("sandwich", "salad", "pasta", "burger",
                    "coffee", "tea", "water", "juice", "wine", "beer", "bottle")
                foodItems.forEach { item ->
                    if (lowerText.contains(item)) {
                        entities["item"] = item
                    }
                }
            }

            "wake_up_call" -> {
                val timeRegex = """(\d{1,2}):?(\d{2})?\s*(am|pm|a\.m\.|p\.m\.)?""".toRegex()
                timeRegex.find(lowerText)?.let {
                    entities["time"] = it.value.trim()
                }

                when {
                    lowerText.contains("morning") -> entities["time"] = "7:00 AM"
                    lowerText.contains("noon") -> entities["time"] = "12:00 PM"
                    lowerText.contains("afternoon") -> entities["time"] = "2:00 PM"
                    lowerText.contains("evening") -> entities["time"] = "6:00 PM"
                }
            }

            "temperature_control" -> {
                when {
                    lowerText.contains("hot") || lowerText.contains("warm") ||
                            lowerText.contains("cool") || lowerText.contains("ac") ||
                            lowerText.contains("air conditioning") || lowerText.contains("cooler") ->
                        entities["action"] = "cooling"

                    lowerText.contains("cold") || lowerText.contains("heat") ||
                            lowerText.contains("heating") || lowerText.contains("warmer") ->
                        entities["action"] = "heating"
                }

                val tempRegex = """(\d+)\s*degrees?""".toRegex()
                tempRegex.find(lowerText)?.let {
                    entities["temperature"] = it.groupValues[1].toInt()
                }
            }

            "lighting_control" -> {
                when {
                    lowerText.contains("on") || lowerText.contains("turn on") ||
                            lowerText.contains("switch on") || lowerText.contains("lights on") ->
                        entities["action"] = "on"

                    lowerText.contains("off") || lowerText.contains("turn off") ||
                            lowerText.contains("switch off") || lowerText.contains("lights off") ->
                        entities["action"] = "off"

                    lowerText.contains("dim") || lowerText.contains("darker") ||
                            lowerText.contains("lower") ->
                        entities["action"] = "dim"

                    lowerText.contains("bright") || lowerText.contains("brighter") ||
                            lowerText.contains("increase") ->
                        entities["action"] = "brighten"
                }
            }

            "room_cleaning" -> {
                when {
                    lowerText.contains("now") || lowerText.contains("immediately") ->
                        entities["urgency"] = "high"
                    lowerText.contains("later") || lowerText.contains("afternoon") ->
                        entities["urgency"] = "low"
                }
            }

            "maintenance" -> {
                when {
                    lowerText.contains("broken") || lowerText.contains("not working") ->
                        entities["status"] = "broken"
                    lowerText.contains("leak") -> entities["issue"] = "leak"
                    lowerText.contains("noise") -> entities["issue"] = "noise"
                    lowerText.contains("light") -> entities["area"] = "lighting"
                    lowerText.contains("ac") || lowerText.contains("air conditioning") ->
                        entities["area"] = "hvac"
                }
            }
        }

        return entities
    }

    fun release() {
        try {
            interpreter.close()
            Log.d(TAG, "📚 NLU service released")
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing NLU: ${e.message}", e)
        }
    }
}