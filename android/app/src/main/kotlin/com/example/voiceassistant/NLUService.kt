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

    // Refined keyword dictionary — ambiguous single words removed to prevent false positives
    // (e.g. "water" matched "water polo" as food_order). Ambiguous cases fall to ML model.
    private val intentDictionary = mapOf(
        "food_order" to listOf(
            "bottled water", "glass of water", "drinking water", "bottle of water", "water bottle",
            "drink", "beverage", "sandwich", "food", "burger", "pizza", "coffee", "tea",
            "menu", "order food", "order room service", "place an order",
            "eat", "hungry", "breakfast", "lunch", "dinner", "juice", "coke", "snack",
            "meal", "restaurant", "dining", "ice bucket", "ice cubes", "fruit",
            "wine", "beer", "champagne", "room service", "kitchen"
        ),
        "room_cleaning" to listOf(
            "housekeeping", "clean my room", "clean the room", "room cleaning",
            "clean my bathroom", "clean the bathroom", "bathroom cleaned", "bathroom clean",
            "tidy", "tidy up", "sweep", "mop", "maid", "maid service",
            "dust", "trash", "garbage", "bin", "vacuum", "turndown", "turndown service"
        ),
        "towel_request" to listOf(
            "towel", "towels", "bath towel", "hand towel", "face towel", "washcloth", "bath mat"
        ),
        "toiletries_request" to listOf(
            "toiletries", "soap", "shampoo", "toothpaste", "toothbrush", "dental",
            "shaving", "razor", "comb", "lotion", "conditioner", "body wash",
            "tissues", "toilet paper", "toilet roll"
        ),
        "maintenance" to listOf(
            "maintenance", "broken", "fix", "repair", "light bulb", "leak", "drain", "clogged",
            "not working", "ac not working", "air conditioning broken",
            "shower not working", "faucet", "toilet broken", "toilet not working",
            "tv not working", "remote not working", "outlet",
            "door lock", "lock not working", "window broken"
        ),
        "concierge_taxi" to listOf(
            "taxi", "cab", "uber", "transport", "airport shuttle", "limo",
            "car service", "rental car", "driver", "book a taxi", "call a cab"
        ),
        "wake_up_call" to listOf(
            "wake up", "alarm", "morning call", "wake me up"
        ),
        "checkout_billing" to listOf(
            "bill", "checkout", "check out", "leaving", "invoice", "receipt", "my bill", "folio"
        ),
        "pillow_request" to listOf(
            "pillow", "pillows", "extra pillow", "cushion"
        ),
        "blanket_request" to listOf(
            "blanket", "blankets", "extra blanket", "duvet", "comforter", "linen"
        ),
        "laundry_service" to listOf(
            "laundry", "dry clean", "ironing", "wash clothes", "wash my clothes", "laundry service"
        ),
        "noise_complaint" to listOf(
            "noise", "too loud", "noisy", "neighbor", "noise complaint", "barking", "keep it down"
        ),
        "concierge_general" to listOf(
            "wifi", "internet", "wifi password", "swimming pool", "pool hours",
            "map", "tour", "recommendation", "dinner booking", "reservation",
            "ticket", "event", "attraction", "gym", "spa"
        ),
        "do_not_disturb" to listOf(
            "do not disturb", "dnd", "privacy sign", "disturb"
        ),
        "emergency" to listOf(
            "emergency", "need help", "help me", "doctor", "medical", "police",
            "fire alarm", "accident", "hurt", "sick", "ambulance", "urgent"
        ),
        "lighting_control" to listOf(
            "lights", "lamp", "turn on lights", "turn off lights", "brighten", "dim the lights"
        ),
        "temperature_control" to listOf(
            "temperature", "thermostat", "warmer", "cooler", "air conditioning", "ac",
            "turn on ac", "turn off ac"
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