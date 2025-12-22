package com.example.voiceassistant

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.json.JSONObject
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

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

    // Comprehensive keyword dictionary for hotel requests
    private val intentDictionary = mapOf(
        "food_order" to listOf(
            "water", "bottle", "drink", "beverage", "sandwich", "food", "burger", "pizza", "coffee", "tea", 
            "menu", "order", "eat", "hungry", "breakfast", "lunch", "dinner", "juice", "coke", "snack", 
            "meal", "restaurant", "dining", "ice", "fruit", "wine", "beer", "champagne", "kitchen"
        ),
        "room_cleaning" to listOf(
            "housekeeping", "clean", "cleaning", "tidy", "makeup", "sweep", "mop", "maid", "maid service",
            "dust", "trash", "garbage", "bin", "vacuum", "turndown"
        ),
        "towel_request" to listOf(
            "towel", "towels", "bath towel", "hand towel", "face towel", "washcloth", "bath mat"
        ),
        "toiletries_request" to listOf(
            "toiletries", "soap", "shampoo", "toothpaste", "toothbrush", "dental", "kit", "shaving", 
            "razor", "comb", "lotion", "gel", "conditioner", "body wash", "tissues", "toilet paper", "toilet roll"
        ),
        "maintenance" to listOf(
            "maintenance", "broken", "fix", "repair", "light bulb", "leak", "drain", "clogged", 
            "not working", "ac not working", "air conditioning", "shower", "faucet", "toilet", "tv", 
            "remote", "outlet", "power", "switch", "door", "lock", "window"
        ),
        "concierge_taxi" to listOf(
            "taxi", "cab", "uber", "ride", "transport", "airport shuttle", "limo", "car", "driver"
        ),
        "wake_up_call" to listOf(
            "wake up", "alarm", "morning call", "wake me up"
        ),
        "checkout_billing" to listOf(
            "bill", "checkout", "check out", "leaving", "invoice", "receipt", "pay", "account", "folio"
        ),
        "pillow_request" to listOf(
            "pillow", "pillows", "extra pillow", "cushion"
        ),
        "blanket_request" to listOf(
            "blanket", "blankets", "extra blanket", "duvet", "comforter", "sheet", "linen"
        ),
        "laundry_service" to listOf(
            "laundry", "wash", "dry clean", "ironing", "pressing"
        ),
        "noise_complaint" to listOf(
            "noise", "loud", "noisy", "quiet", "neighbor", "party", "barking"
        ),
        "concierge_general" to listOf(
            "wifi", "internet", "password", "connection", "area", "map", "tour", "recommendation", 
            "dinner booking", "reservation", "ticket", "event", "attraction", "gym", "pool", "spa"
        ),
        "do_not_disturb" to listOf(
            "disturb", "dnd", "privacy", "privacy sign", "do not disturb"
        ),
        "emergency" to listOf(
            "emergency", "help", "doctor", "medical", "police", "fire", "accident", "hurt", "sick", "ambulance"
        ),
        "lighting_control" to listOf(
            "lights", "lamp", "dim", "brighten", "turn on lights", "turn off lights"
        ),
        "temperature_control" to listOf(
            "temperature", "thermostat", "warmer", "cooler", "heat", "ac", "fan"
        )
    )

    // Optimization: Pre-compile Regex patterns to avoid overhead during inference
    private val compiledRules: List<Pair<String, List<Regex>>> by lazy {
        intentDictionary.map { (intent, keywords) ->
            intent to keywords.map { keyword -> 
                Regex("\\b${Regex.escape(keyword)}\\b", RegexOption.IGNORE_CASE) 
            }
        }
    }

    fun initialize() {
        try {
            Log.d(TAG, "🔧 Initializing NLU Service...")

            // Load TFLite model
            val modelFile = loadModelFile("models/nlu/hotel_mobilebert.tflite")
            interpreter = Interpreter(modelFile)

            // Load vocabulary
            val vocabJson = context.assets.open("models/nlu/vocab.json")
                .bufferedReader().use { it.readText() }
            vocab = JSONObject(vocabJson).let { json ->
                json.keys().asSequence().associateWith { json.getInt(it) }
            }

            // Load label mapping
            val labelJson = context.assets.open("models/nlu/label_map.json")
                .bufferedReader().use { it.readText() }
            labelMap = JSONObject(labelJson).let { json ->
                json.keys().asSequence().associate { it.toInt() to json.getString(it) }
            }

            Log.d(TAG, "✅ NLU initialized")
            runInitialTests()

        } catch (e: Exception) {
            Log.e(TAG, "❌ NLU init failed: ${e.message}", e)
        }
    }

    private fun runInitialTests() {
        val tests = mapOf(
            "i need a towel" to "towel_request",
            "i need a sandwich" to "food_order",
            "what is the wifi password" to "concierge_general"
        )
        
        Log.d(TAG, "🧪 Running NLU tests...")
        var passed = 0
        tests.forEach { (input, expected) ->
            if (classifyIntent(input).name == expected) passed++
        }
        Log.d(TAG, "📊 Test results: $passed/${tests.size} passed")
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
        val lowerText = text.lowercase().trim()
        
        // 1. Check Rule-Based Overrides (Fast Path)
        val ruleResult = checkRules(lowerText)
        if (ruleResult != null) {
            Log.d(TAG, "🎯 Rule-based match: ${ruleResult.name}")
            return ruleResult
        }

        try {
            // 2. Fallback to TFLite Model (Slow Path)
            val tokens = tokenize(lowerText)
            val inputArray = Array(1) { IntArray(MAX_SEQ_LENGTH) }
            tokens.forEachIndexed { index, token ->
                if (index < MAX_SEQ_LENGTH) inputArray[0][index] = token
            }

            val outputArray = Array(1) { FloatArray(labelMap.size) }
            interpreter.run(inputArray, outputArray)

            val scores = outputArray[0]
            val probabilities = softmax(scores)
            val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
            val confidence = probabilities[maxIndex]
            val intentName = labelMap[maxIndex] ?: "misc_request"

            return Intent(intentName, confidence, extractEntities(lowerText, intentName))

        } catch (e: Exception) {
            Log.e(TAG, "❌ Model failed: ${e.message}")
            return Intent("misc_request", 0f)
        }
    }

    private fun checkRules(text: String): Intent? {
        for ((intent, patterns) in compiledRules) {
            for (pattern in patterns) {
                if (pattern.containsMatchIn(text)) {
                    return Intent(intent, 0.99f, extractEntities(text, intent))
                }
            }
        }
        return null
    }

    private fun tokenize(text: String): List<Int> {
        val tokens = mutableListOf<Int>()
        tokens.add(vocab[CLS_TOKEN] ?: 101)
        text.split(Regex("\\s+")).take(MAX_SEQ_LENGTH - 2).forEach { word ->
            tokens.add(vocab[word] ?: vocab[UNK_TOKEN] ?: 100)
        }
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
        val quantityRegex = """(\d+)|one|two|three|four|five|six|seven|eight|nine|ten""".toRegex()
        quantityRegex.find(text)?.let { entities["quantity"] = it.value }
        return entities
    }

    fun close() {
        if (::interpreter.isInitialized) interpreter.close()
    }
}