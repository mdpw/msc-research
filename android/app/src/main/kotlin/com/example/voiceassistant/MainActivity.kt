package com.example.voiceassistant

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.*
import com.example.voiceassistant.ui.theme.VoiceAssistantTheme
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.platform.LocalConfiguration
import android.content.res.Configuration
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.os.Build
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull
import java.util.*

data class RequestItem(
    val id: Int,
    val requestText: String,
    val intent: String,
    val confidence: Float,
    val department: String,
    val status: String,
    val timestamp: String,
    val rating: Int = 0
)

// Theme colors defined in ui/theme/Theme.kt (SeraLightColors, SeraDarkColors)

// ── Department Colors ──
fun getDepartmentColor(department: String): Color = when (department) {
    "Housekeeping" -> Color(0xFF4CAF50)
    "Room Service" -> Color(0xFFFF9800)
    "Maintenance" -> Color(0xFF2196F3)
    "Front Desk" -> Color(0xFF9C27B0)
    "Concierge" -> Color(0xFF00BCD4)
    else -> Color(0xFF757575)
}

fun getStatusBorderColor(status: String): Color = when (status) {
    "pending" -> Color(0xFFFFC107)
    "in_progress" -> Color(0xFF2196F3)
    "completed" -> Color(0xFF4CAF50)
    "cancelled" -> Color(0xFF9E9E9E)
    else -> Color.Gray
}

// ── Relative Time Helper ──
fun getRelativeTime(timestamp: String): String {
    try {
        if (timestamp.contains("T") || timestamp.contains("-")) {
            val parts = timestamp.replace("T", " ").split(" ")
            if (parts.size >= 2) {
                val dateParts = parts[0].split("-")
                val timeParts = parts[1].split(":")
                if (dateParts.size == 3 && timeParts.size >= 2) {
                    val cal = Calendar.getInstance()
                    cal.set(dateParts[0].toInt(), dateParts[1].toInt() - 1, dateParts[2].toInt(),
                        timeParts[0].toInt(), timeParts[1].toInt(),
                        if (timeParts.size > 2) timeParts[2].split(".")[0].toInt() else 0)
                    val diffMs = System.currentTimeMillis() - cal.timeInMillis
                    val diffMin = diffMs / 60000
                    return when {
                        diffMin < 1 -> "Just now"
                        diffMin < 60 -> "${diffMin}m ago"
                        diffMin < 1440 -> "${diffMin / 60}h ago"
                        else -> "${diffMin / 1440}d ago"
                    }
                }
            }
        }
    } catch (_: Exception) {}
    return timestamp
}

// ── Confirmation state for Feature 3 ──
data class PendingConfirmation(
    val transcription: String,
    val intentName: String,
    val confidence: Float
)

class MainActivity : ComponentActivity() {
    private lateinit var voskService: VoskService
    private lateinit var audioRecorder: AudioRecorder
    private lateinit var nluService: NLUService
    private val _apiService = mutableStateOf<ApiService?>(null)
    private var apiService: ApiService
        get() = _apiService.value!!
        set(value) { _apiService.value = value }
    private lateinit var tts: TextToSpeech
    private lateinit var webSocketService: WebSocketService
    private var ttsReady = false
    private val pendingTtsQueue = mutableListOf<Pair<String, String>>()
    private val ttsReadyDeferred = CompletableDeferred<Unit>()
    private val TAG = "MainActivity"
    private val ROOM_NUMBER = "101"
    private val _requestHistory = mutableStateListOf<RequestItem>()

    // Network & server config state
    private val _wifiSsid = mutableStateOf<String?>(null)
    private val _deviceIp = mutableStateOf<String?>(null)
    private val _serverIp = mutableStateOf("192.168.1.100")
    private val _serverPort = mutableIntStateOf(8000)

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            initializeServices()
        } else {
            Toast.makeText(this, "Microphone permission is required.", Toast.LENGTH_LONG).show()
        }
    }

    private val locationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            refreshNetworkInfo()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        ServerConfig.load(this)
        _serverIp.value = ServerConfig.serverIp
        _serverPort.intValue = ServerConfig.serverPort

        voskService = VoskService(this)
        audioRecorder = AudioRecorder(this)
        nluService = NLUService(this)
        apiService = ApiService(ServerConfig.baseUrl)

        tts = TextToSpeech(this) { status ->
            Log.d(TAG, "TTS init status: $status (SUCCESS=${TextToSpeech.SUCCESS})")
            if (status == TextToSpeech.SUCCESS) {
                val result = tts.setLanguage(Locale.US)
                Log.d(TAG, "TTS setLanguage(US) result: $result")
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.e(TAG, "TTS: US English not available, trying UK")
                    val resultUK = tts.setLanguage(Locale.UK)
                    if (resultUK == TextToSpeech.LANG_MISSING_DATA || resultUK == TextToSpeech.LANG_NOT_SUPPORTED) {
                        Log.e(TAG, "TTS: UK English not available, trying ENGLISH")
                        tts.setLanguage(Locale.ENGLISH)
                    }
                }
                // Set ready regardless — some devices return LANG_MISSING_DATA but still speak
                tts.setSpeechRate(0.9f)
                tts.setPitch(1.0f)
                ttsReady = true
                Log.d(TAG, "TTS ready = true, engine: ${tts.defaultEngine}")
            } else {
                Log.e(TAG, "TTS init FAILED with status: $status")
            }
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            initializeServices()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
            refreshNetworkInfo()
        } else {
            locationPermissionLauncher.launch(Manifest.permission.ACCESS_FINE_LOCATION)
        }

        connectWebSocket()

        setContent {
            VoiceAssistantScreen(
                roomNumber = ROOM_NUMBER,
                voskService = voskService,
                audioRecorder = audioRecorder,
                nluService = nluService,
                apiService = _apiService.value ?: return@setContent,
                lifecycleScope = lifecycleScope,
                requestHistory = _requestHistory,
                wifiSsid = _wifiSsid,
                deviceIp = _deviceIp,
                serverIp = _serverIp.value,
                serverPort = _serverPort.intValue,
                onSpeakAndWait = { message -> speakAndWait(message) },
                onSpeakResponse = { message -> speakFire(message) },
                onAddRequest = { request -> _requestHistory.add(0, request) },
                onCloseApp = { finish() },
                onRefreshRequests = { refreshRequests() },
                onServerUpdate = { ip, port -> updateServer(ip, port) },
                onCancelRequest = { requestId -> cancelRequest(requestId) },
                onRateRequest = { requestId, rating -> rateRequest(requestId, rating) }
            )
        }

        refreshRequests()
    }

    private fun refreshRequests() {
        apiService.getRequestHistory(ROOM_NUMBER,
            onSuccess = { history ->
                runOnUiThread {
                    _requestHistory.clear()
                    _requestHistory.addAll(history)
                }
            },
            onError = { error ->
                runOnUiThread {
                    Toast.makeText(this, "Sync failed", Toast.LENGTH_SHORT).show()
                    speakFire("There is an issue with network connectivity. Please contact the help desk using the land line phone.")
                }
            }
        )
    }

    private fun initializeServices() {
        lifecycleScope.launch { voskService.initialize() }
        nluService.initialize()
    }

    private fun refreshNetworkInfo() {
        _wifiSsid.value = NetworkUtils.getWifiSsid(this)
        _deviceIp.value = NetworkUtils.getDeviceIp(this)
    }

    private fun updateServer(ip: String, port: Int) {
        ServerConfig.saveServer(this, ip, port)
        _serverIp.value = ip
        _serverPort.intValue = port
        reconnectToServer()
    }

    // Feature 5: Cancel request
    private fun cancelRequest(requestId: Int) {
        apiService.cancelRequest(requestId, ROOM_NUMBER,
            onSuccess = {
                runOnUiThread {
                    val index = _requestHistory.indexOfFirst { it.id == requestId }
                    if (index != -1) {
                        _requestHistory[index] = _requestHistory[index].copy(status = "cancelled")
                    }
                    speakFire("Your request number $requestId has been cancelled.")
                }
            },
            onError = { error ->
                runOnUiThread {
                    Toast.makeText(this, "Cannot cancel: $error", Toast.LENGTH_SHORT).show()
                }
            }
        )
    }

    // Feature 10: Rate request
    private fun rateRequest(requestId: Int, rating: Int) {
        apiService.rateRequest(requestId, ROOM_NUMBER, rating,
            onSuccess = {
                runOnUiThread {
                    val index = _requestHistory.indexOfFirst { it.id == requestId }
                    if (index != -1) {
                        _requestHistory[index] = _requestHistory[index].copy(rating = rating)
                    }
                    speakFire("Thank you for your feedback!")
                }
            },
            onError = { error ->
                runOnUiThread {
                    Toast.makeText(this, "Rating failed: $error", Toast.LENGTH_SHORT).show()
                }
            }
        )
    }

    private fun requestAudioFocus() {
        val audioManager = getSystemService(AUDIO_SERVICE) as AudioManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val focusRequest = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_ASSISTANT)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build()
                )
                .build()
            audioManager.requestAudioFocus(focusRequest)
        } else {
            @Suppress("DEPRECATION")
            audioManager.requestAudioFocus(null, AudioManager.STREAM_MUSIC, AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
        }
    }

    private fun flushPendingTtsQueue() {
        if (!ttsReady) return
        pendingTtsQueue.forEach { (message, utteranceId) ->
            val params = Bundle().apply {
                putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, utteranceId)
            }
            tts.speak(message, TextToSpeech.QUEUE_ADD, params, utteranceId)
        }
        pendingTtsQueue.clear()
    }

    private fun speakFire(message: String) {
        if (!ttsReady) {
            pendingTtsQueue.add(message to "fire")
            return
        }
        requestAudioFocus()
        val params = Bundle()
        params.putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, "fire")
        tts.speak(message, TextToSpeech.QUEUE_FLUSH, params, "fire")
    }

    private suspend fun speakAndWait(message: String) {
        if (!ttsReady) {
            withTimeoutOrNull(5000L) { ttsReadyDeferred.await() }
        }
        if (!ttsReady) {
            pendingTtsQueue.add(message to "greeting")
            return
        }
        requestAudioFocus()
        val deferred = CompletableDeferred<Unit>()
        tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {}
            override fun onDone(utteranceId: String?) { deferred.complete(Unit) }
            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) { deferred.complete(Unit) }
        })
        val params = Bundle()
        params.putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, "greeting")
        tts.speak(message, TextToSpeech.QUEUE_FLUSH, params, "greeting")
        withTimeoutOrNull(10_000L) { deferred.await() }
    }

    private fun reconnectToServer() {
        apiService = ApiService(ServerConfig.baseUrl)
        if (::webSocketService.isInitialized) webSocketService.disconnect()
        connectWebSocket()
        apiService.getRequestHistory(ROOM_NUMBER,
            onSuccess = { history ->
                runOnUiThread {
                    _requestHistory.clear()
                    _requestHistory.addAll(history)
                    speakFire("Successfully connected to server. System is ready.")
                }
            },
            onError = {
                runOnUiThread {
                    speakFire("Failed to connect to server at ${ServerConfig.serverIp}. Please check the network connection or contact the help desk using the land line phone.")
                }
            }
        )
    }

    private fun connectWebSocket() {
        try {
            webSocketService = WebSocketService(ROOM_NUMBER, ServerConfig.wsUrl(ROOM_NUMBER))
            webSocketService.connect(
                onMessage = { message ->
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, message, Toast.LENGTH_LONG).show()
                    }
                },
                onStatusChange = { requestId, status, backendMessage ->
                    runOnUiThread {
                        val index = _requestHistory.indexOfFirst { it.id == requestId }
                        if (index != -1) {
                            _requestHistory[index] = _requestHistory[index].copy(status = status)
                        }

                        val statusMessage = if (backendMessage.isNotBlank()) {
                            backendMessage
                        } else {
                            when (status) {
                                "in_progress" -> "Your request No.$requestId is now being processed."
                                "completed" -> "Your request No.$requestId is completed. Would you like to rate the service?"
                                else -> "Your request No.$requestId is now $status."
                            }
                        }

                        speakFire(statusMessage)

                        if (index == -1) {
                            refreshRequests()
                        }
                    }
                },
                // Feature 9: Staff message handler
                onStaffMsg = { requestId, message, staffName ->
                    runOnUiThread {
                        speakFire("Message from hotel staff regarding your request number $requestId: $message")
                    }
                }
            )
        } catch (e: Exception) {
            Log.e(TAG, "WebSocket error", e)
            runOnUiThread {
                speakFire("There is an issue with network connectivity. Please contact the help desk using the land line phone.")
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        tts.shutdown()
        voskService.release()
        nluService.close()
        webSocketService.disconnect()
    }
}

// ── Animated Mic Button ──
@Composable
fun AnimatedMicButton(
    isRecording: Boolean,
    isProcessing: Boolean,
    audioLevel: Float,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val infiniteTransition = rememberInfiniteTransition(label = "mic")

    val pulseScale by infiniteTransition.animateFloat(
        initialValue = 1f, targetValue = 1.15f,
        animationSpec = infiniteRepeatable(tween(600, easing = EaseInOut), RepeatMode.Reverse),
        label = "pulse"
    )
    val spinAngle by infiniteTransition.animateFloat(
        initialValue = 0f, targetValue = 360f,
        animationSpec = infiniteRepeatable(tween(1000, easing = LinearEasing)),
        label = "spin"
    )
    val glowAlpha by infiniteTransition.animateFloat(
        initialValue = 0.2f, targetValue = 0.6f,
        animationSpec = infiniteRepeatable(tween(800, easing = EaseInOut), RepeatMode.Reverse),
        label = "glow"
    )

    val currentScale = if (isRecording) pulseScale else 1f
    val buttonColor = when {
        isRecording -> MaterialTheme.colorScheme.error
        isProcessing -> MaterialTheme.colorScheme.tertiary
        else -> MaterialTheme.colorScheme.primary
    }
    val glowColor = buttonColor

    Box(contentAlignment = Alignment.Center, modifier = modifier.size(120.dp)) {
        if (isRecording) {
            val normalizedLevel = (audioLevel * 8f).coerceIn(0f, 1f)
            Canvas(modifier = Modifier.size(120.dp)) {
                val ringRadius = size.minDimension / 2f
                drawCircle(glowColor.copy(alpha = glowAlpha * normalizedLevel), ringRadius * (1f + normalizedLevel * 0.3f), style = Stroke(3.dp.toPx()))
                drawCircle(glowColor.copy(alpha = glowAlpha * normalizedLevel * 0.5f), ringRadius * (1f + normalizedLevel * 0.5f), style = Stroke(2.dp.toPx()))
            }
        }
        if (isProcessing) {
            Canvas(modifier = Modifier.size(100.dp)) {
                drawArc(glowColor.copy(alpha = 0.7f), spinAngle, 120f, false, Offset.Zero, Size(size.width, size.height), style = Stroke(3.dp.toPx(), cap = StrokeCap.Round))
            }
        }
        Surface(
            onClick = { if (!isRecording && !isProcessing) onClick() },
            modifier = Modifier.size(88.dp).scale(currentScale).shadow(if (isRecording) 12.dp else 6.dp, CircleShape),
            shape = CircleShape, color = buttonColor
        ) {
            Box(contentAlignment = Alignment.Center, modifier = Modifier.fillMaxSize()) {
                Canvas(modifier = Modifier.size(36.dp)) {
                    val w = size.width; val h = size.height; val iconColor = Color.White; val strokeW = 2.5.dp.toPx()
                    if (isRecording) {
                        drawRoundRect(iconColor, Offset(w * 0.25f, h * 0.25f), Size(w * 0.5f, h * 0.5f), androidx.compose.ui.geometry.CornerRadius(4.dp.toPx()))
                    } else if (isProcessing) {
                        for (i in 0 until 3) drawCircle(iconColor, 3.dp.toPx(), Offset(w * (0.25f + i * 0.25f), h * 0.5f))
                    } else {
                        drawRoundRect(iconColor, Offset(w * 0.35f, h * 0.1f), Size(w * 0.3f, h * 0.45f), androidx.compose.ui.geometry.CornerRadius(w * 0.15f))
                        drawArc(iconColor, 0f, 180f, false, Offset(w * 0.22f, h * 0.2f), Size(w * 0.56f, h * 0.55f), style = Stroke(strokeW, cap = StrokeCap.Round))
                        drawLine(iconColor, Offset(w * 0.5f, h * 0.75f), Offset(w * 0.5f, h * 0.88f), strokeW, StrokeCap.Round)
                        drawLine(iconColor, Offset(w * 0.35f, h * 0.88f), Offset(w * 0.65f, h * 0.88f), strokeW, StrokeCap.Round)
                    }
                }
            }
        }
    }
}

// ── Audio Level Indicator ──
@Composable
fun AudioLevelIndicator(audioLevel: Float, isRecording: Boolean, modifier: Modifier = Modifier) {
    if (!isRecording) return
    val normalizedLevel = (audioLevel * 10f).coerceIn(0f, 1f)
    Row(modifier = modifier.height(24.dp), horizontalArrangement = Arrangement.spacedBy(2.dp), verticalAlignment = Alignment.CenterVertically) {
        val barCount = 20
        for (i in 0 until barCount) {
            val threshold = i.toFloat() / barCount
            val isActive = normalizedLevel > threshold
            Box(modifier = Modifier.weight(1f).fillMaxHeight(if (isActive) 0.3f + (normalizedLevel - threshold).coerceIn(0f, 0.7f) else 0.2f).clip(RoundedCornerShape(2.dp)).background(
                if (isActive) when { i < barCount * 0.6 -> Color(0xFF4CAF50); i < barCount * 0.8 -> Color(0xFFFFC107); else -> Color(0xFFD32F2F) } else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.3f)
            ))
        }
    }
}

@Composable
fun VoiceAssistantScreen(
    roomNumber: String,
    voskService: VoskService,
    audioRecorder: AudioRecorder,
    nluService: NLUService,
    apiService: ApiService,
    lifecycleScope: kotlinx.coroutines.CoroutineScope,
    requestHistory: List<RequestItem>,
    wifiSsid: State<String?>,
    deviceIp: State<String?>,
    serverIp: String,
    serverPort: Int,
    onSpeakAndWait: suspend (String) -> Unit,
    onSpeakResponse: (String) -> Unit,
    onAddRequest: (RequestItem) -> Unit,
    onCloseApp: () -> Unit,
    onRefreshRequests: () -> Unit,
    onServerUpdate: (String, Int) -> Unit,
    onCancelRequest: (Int) -> Unit,
    onRateRequest: (Int, Int) -> Unit
) {
    val context = LocalContext.current
    var isRecording by remember { mutableStateOf(false) }
    var isProcessing by remember { mutableStateOf(false) }
    var statusMessage by remember { mutableStateOf("Tap microphone to start") }
    var lastTranscription by remember { mutableStateOf("") }
    var lastIntent by remember { mutableStateOf("") }
    var lastConfidence by remember { mutableStateOf(0f) }
    var lastDepartment by remember { mutableStateOf("") }
    var showServerDialog by remember { mutableStateOf(false) }
    var settingsExpanded by remember { mutableStateOf(false) }
    var audioLevel by remember { mutableFloatStateOf(0f) }

    // Feature 3: Confirmation state
    var pendingConfirmation by remember { mutableStateOf<PendingConfirmation?>(null) }

    // Feature 10: Rating dialog state
    var ratingRequestId by remember { mutableIntStateOf(-1) }

    val sortedRequests = requestHistory.sortedWith(compareBy {
        when (it.status) {
            "in_progress" -> 0
            "pending" -> 1
            "completed" -> 2
            "cancelled" -> 3
            else -> 4
        }
    })

    val isDark = isSystemInDarkTheme()

    val isLandscape = LocalConfiguration.current.orientation == Configuration.ORIENTATION_LANDSCAPE

    VoiceAssistantTheme(darkTheme = isDark) {
        Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            Box(modifier = Modifier.fillMaxSize()) {
                if (isLandscape) {
                    // ── LANDSCAPE: two-column layout ──
                    LandscapeLayout(
                        roomNumber = roomNumber, isDark = isDark,
                        settingsExpanded = settingsExpanded, onToggleSettings = { settingsExpanded = !settingsExpanded },
                        wifiSsid = wifiSsid, deviceIp = deviceIp, serverIp = serverIp, serverPort = serverPort,
                        onEditServer = { showServerDialog = true }, onCloseApp = onCloseApp,
                        pendingConfirmation = pendingConfirmation,
                        onConfirmSubmit = {
                            val confirmation = pendingConfirmation!!
                            pendingConfirmation = null; statusMessage = "Submitting..."; isProcessing = true
                            val handler = android.os.Handler(android.os.Looper.getMainLooper())
                            apiService.submitRequest(
                                roomNumber = roomNumber, requestText = confirmation.transcription, intent = confirmation.intentName,
                                onSuccess = { response -> handler.post { val request = RequestItem(response.requestId, confirmation.transcription, confirmation.intentName, confirmation.confidence, "Routing...", "pending", getCurrentTime()); lastDepartment = request.department; onAddRequest(request); onSpeakResponse("Done! Your request number ${response.requestId} has been sent to the ${response.department} team."); isProcessing = false; statusMessage = "Tap microphone to start"; onRefreshRequests() } },
                                onError = { handler.post { onSpeakResponse("Sorry, I could not send your request due to a network issue. Please contact the help desk using the land line phone."); isProcessing = false; statusMessage = "Submission failed" } }
                            )
                        },
                        onConfirmCancel = { pendingConfirmation = null; statusMessage = "Request cancelled"; onSpeakResponse("Okay, I've cancelled that request.") },
                        isRecording = isRecording, isProcessing = isProcessing, audioLevel = audioLevel,
                        onMicClick = {
                            if (!isRecording && !isProcessing) {
                                lifecycleScope.launch {
                                    statusMessage = "Hello! I'm Sera. How can I help you?"
                                    onSpeakAndWait("Hello! I'm Sera. How can I help you?")
                                    processVoiceRequest(audioRecorder, voskService, nluService, apiService, roomNumber, lifecycleScope, { isRecording = true }, { isRecording = false }, { isProcessing = true }, { isProcessing = false }, { statusMessage = it }, { lastTranscription = it }, { i, c -> lastIntent = i; lastConfidence = c; Toast.makeText(context, "Intent: $i (${(c * 100).toInt()}%)", Toast.LENGTH_SHORT).show() }, { level -> audioLevel = level }, onSpeakResponse, onSpeakAndWait, { request -> lastDepartment = request.department; onAddRequest(request) }, onRefreshRequests, { statusMessage = "Tap microphone to start"; audioLevel = 0f }, { transcription, intentName, confidence -> pendingConfirmation = PendingConfirmation(transcription, intentName, confidence); lastTranscription = transcription; lastIntent = intentName; lastConfidence = confidence; statusMessage = "Tap Yes or No"; isProcessing = false }, onCancelRequest, requestHistory.filter { it.status in listOf("pending", "in_progress") })
                                }
                            }
                        },
                        lastTranscription = lastTranscription, lastDepartment = lastDepartment,
                        sortedRequests = sortedRequests, onRefreshRequests = onRefreshRequests,
                        onCancelRequest = onCancelRequest, onRate = { ratingRequestId = it }
                    )
                } else {
                    // ── PORTRAIT: vertical stacked layout ──
                    PortraitLayout(
                        roomNumber = roomNumber, isDark = isDark,
                        settingsExpanded = settingsExpanded, onToggleSettings = { settingsExpanded = !settingsExpanded },
                        wifiSsid = wifiSsid, deviceIp = deviceIp, serverIp = serverIp, serverPort = serverPort,
                        onEditServer = { showServerDialog = true }, onCloseApp = onCloseApp,
                        pendingConfirmation = pendingConfirmation,
                        onConfirmSubmit = {
                            val confirmation = pendingConfirmation!!
                            pendingConfirmation = null; statusMessage = "Submitting..."; isProcessing = true
                            val handler = android.os.Handler(android.os.Looper.getMainLooper())
                            apiService.submitRequest(
                                roomNumber = roomNumber, requestText = confirmation.transcription, intent = confirmation.intentName,
                                onSuccess = { response -> handler.post { val request = RequestItem(response.requestId, confirmation.transcription, confirmation.intentName, confirmation.confidence, "Routing...", "pending", getCurrentTime()); lastDepartment = request.department; onAddRequest(request); onSpeakResponse("Done! Your request number ${response.requestId} has been sent to the ${response.department} team."); isProcessing = false; statusMessage = "Tap microphone to start"; onRefreshRequests() } },
                                onError = { handler.post { onSpeakResponse("Sorry, I could not send your request due to a network issue. Please contact the help desk using the land line phone."); isProcessing = false; statusMessage = "Submission failed" } }
                            )
                        },
                        onConfirmCancel = { pendingConfirmation = null; statusMessage = "Request cancelled"; onSpeakResponse("Okay, I've cancelled that request.") },
                        isRecording = isRecording, isProcessing = isProcessing, audioLevel = audioLevel,
                        onMicClick = {
                            if (!isRecording && !isProcessing) {
                                lifecycleScope.launch {
                                    statusMessage = "Hello! I'm Sera. How can I help you?"
                                    onSpeakAndWait("Hello! I'm Sera. How can I help you?")
                                    processVoiceRequest(audioRecorder, voskService, nluService, apiService, roomNumber, lifecycleScope, { isRecording = true }, { isRecording = false }, { isProcessing = true }, { isProcessing = false }, { statusMessage = it }, { lastTranscription = it }, { i, c -> lastIntent = i; lastConfidence = c; Toast.makeText(context, "Intent: $i (${(c * 100).toInt()}%)", Toast.LENGTH_SHORT).show() }, { level -> audioLevel = level }, onSpeakResponse, onSpeakAndWait, { request -> lastDepartment = request.department; onAddRequest(request) }, onRefreshRequests, { statusMessage = "Tap microphone to start"; audioLevel = 0f }, { transcription, intentName, confidence -> pendingConfirmation = PendingConfirmation(transcription, intentName, confidence); lastTranscription = transcription; lastIntent = intentName; lastConfidence = confidence; statusMessage = "Tap Yes or No"; isProcessing = false }, onCancelRequest, requestHistory.filter { it.status in listOf("pending", "in_progress") })
                                }
                            }
                        },
                        lastTranscription = lastTranscription, lastDepartment = lastDepartment,
                        sortedRequests = sortedRequests, onRefreshRequests = onRefreshRequests,
                        onCancelRequest = onCancelRequest, onRate = { ratingRequestId = it }
                    )
                }
            }
        }

        // Server edit dialog (orientation-independent)
        if (showServerDialog) {
            ServerEditDialog(
                currentIp = serverIp,
                currentPort = serverPort,
                onDismiss = { showServerDialog = false },
                onSave = { ip, port -> onServerUpdate(ip, port); showServerDialog = false }
            )
        }

        // Feature 10: Rating dialog
        if (ratingRequestId > 0) {
            RatingDialog(
                requestId = ratingRequestId,
                onRate = { rating -> onRateRequest(ratingRequestId, rating); ratingRequestId = -1 },
                onDismiss = { ratingRequestId = -1 }
            )
        }
    }
}

// ── Shared composables for both orientations ──

@Composable
private fun HeaderCard(
    roomNumber: String, isDark: Boolean,
    settingsExpanded: Boolean, onToggleSettings: () -> Unit,
    wifiSsid: State<String?>, deviceIp: State<String?>,
    serverIp: String, serverPort: Int,
    onEditServer: () -> Unit, onCloseApp: () -> Unit,
    compact: Boolean = false
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.fillMaxWidth()) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = if (compact) 10.dp else 12.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    if (compact) {
                        Text("Sera", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onPrimaryContainer)
                        Text("Voice Assistant", style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Medium, color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.85f))
                        Text("Room $roomNumber", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f))
                    } else {
                        Text("Sera - Voice Assistant", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onPrimaryContainer)
                        Text("Room $roomNumber", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f))
                    }
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    IconButton(onClick = onToggleSettings, modifier = Modifier.size(32.dp)) {
                        Icon(if (settingsExpanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown, "Toggle Settings", tint = MaterialTheme.colorScheme.onPrimaryContainer)
                    }
                    IconButton(onClick = onCloseApp, modifier = Modifier.size(32.dp)) {
                        Icon(Icons.Default.Close, "Close App", tint = MaterialTheme.colorScheme.error)
                    }
                }
            }
            AnimatedVisibility(visible = settingsExpanded) {
                Card(modifier = Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 4.dp), colors = CardDefaults.cardColors(containerColor = if (isDark) Color(0xFF1B3B2F) else Color(0xFFE8F5E9)), shape = RoundedCornerShape(8.dp)) {
                    Column(modifier = Modifier.fillMaxWidth().padding(8.dp)) {
                        if (compact) {
                            Text("WiFi: ${wifiSsid.value ?: "Not connected"}", style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Medium)
                            Text("IP: ${deviceIp.value ?: "N/A"}", style = MaterialTheme.typography.bodySmall)
                        } else {
                            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                Text("WiFi: ${wifiSsid.value ?: "Not connected"}", style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Medium)
                                Text("Device IP: ${deviceIp.value ?: "N/A"}", style = MaterialTheme.typography.bodySmall)
                            }
                        }
                        Spacer(modifier = Modifier.height(4.dp))
                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                            Text("Server: $serverIp:$serverPort", style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Medium, color = if (isDark) Color(0xFF81C784) else Color(0xFF2E7D32))
                            IconButton(onClick = onEditServer, modifier = Modifier.size(28.dp)) { Icon(Icons.Default.Edit, "Edit Server", modifier = Modifier.size(16.dp)) }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun MicCard(
    pendingConfirmation: PendingConfirmation?,
    onConfirmSubmit: () -> Unit,
    onConfirmCancel: () -> Unit,
    isRecording: Boolean,
    isProcessing: Boolean,
    audioLevel: Float,
    onMicClick: () -> Unit,
    lastTranscription: String,
    lastDepartment: String,
    modifier: Modifier = Modifier,
    narrowButtons: Boolean = false
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.fillMaxSize().padding(10.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
            if (pendingConfirmation != null) {
                val pc = pendingConfirmation
                Text("\"${pc.transcription}\"", fontWeight = FontWeight.Medium, fontSize = 12.sp, maxLines = 3, overflow = TextOverflow.Ellipsis, color = MaterialTheme.colorScheme.onSecondaryContainer)
                Spacer(modifier = Modifier.height(8.dp))
                Text("Submit this request?", style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onSecondaryContainer)
                Spacer(modifier = Modifier.height(8.dp))
                if (narrowButtons) {
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                        Button(onClick = onConfirmSubmit, modifier = Modifier.fillMaxWidth(), colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.secondary)) { Text("Yes, Submit", fontSize = 12.sp) }
                        OutlinedButton(modifier = Modifier.fillMaxWidth(), onClick = onConfirmCancel) { Text("No, Cancel", fontSize = 12.sp) }
                    }
                } else {
                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        Button(onClick = onConfirmSubmit, colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.secondary)) { Text("Yes, Submit") }
                        OutlinedButton(onClick = onConfirmCancel) { Text("No, Cancel") }
                    }
                }
            } else {
                AnimatedMicButton(isRecording = isRecording, isProcessing = isProcessing, audioLevel = audioLevel, onClick = onMicClick)
                Spacer(modifier = Modifier.height(4.dp))
                AudioLevelIndicator(audioLevel, isRecording, Modifier.fillMaxWidth(if (narrowButtons) 0.85f else 0.7f))
                Text(when { isRecording -> "Listening..."; isProcessing -> "Processing..."; else -> "Tap to speak" }, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.7f))
                if (lastTranscription.isNotEmpty()) {
                    Spacer(modifier = Modifier.height(4.dp))
                    HorizontalDivider(color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.2f))
                    Spacer(modifier = Modifier.height(4.dp))
                    Text("\"$lastTranscription\"", fontWeight = FontWeight.Medium, fontSize = if (narrowButtons) 12.sp else 13.sp, maxLines = 2, overflow = TextOverflow.Ellipsis, color = MaterialTheme.colorScheme.onSecondaryContainer)
                    if (lastDepartment.isNotEmpty()) { DepartmentBadge(lastDepartment) }
                }
            }
        }
    }
}

@Composable
private fun RequestsPanel(
    sortedRequests: List<RequestItem>,
    onRefreshRequests: () -> Unit,
    onCancelRequest: (Int) -> Unit,
    onRate: (Int) -> Unit,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        Row(modifier = Modifier.fillMaxWidth().padding(bottom = 4.dp), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
            Text("Recent Requests", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onBackground)
            IconButton(onClick = onRefreshRequests, modifier = Modifier.size(28.dp)) { Icon(Icons.Default.Refresh, "Refresh", modifier = Modifier.size(18.dp), tint = MaterialTheme.colorScheme.onBackground) }
        }
        Box(modifier = Modifier.fillMaxSize()) {
            if (sortedRequests.isEmpty()) {
                Card(modifier = Modifier.fillMaxSize(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant), shape = RoundedCornerShape(12.dp)) {
                    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            val emptyIconColor = MaterialTheme.colorScheme.onSurfaceVariant
                            Canvas(modifier = Modifier.size(56.dp)) {
                                val w = size.width; val h = size.height; val col = emptyIconColor; val sw = 3.dp.toPx()
                                drawRoundRect(col, Offset(w*0.35f,h*0.1f), Size(w*0.3f,h*0.45f), androidx.compose.ui.geometry.CornerRadius(w*0.15f))
                                drawArc(col, 0f, 180f, false, Offset(w*0.2f,h*0.15f), Size(w*0.6f,h*0.55f), style = Stroke(sw, cap = StrokeCap.Round))
                                drawLine(col, Offset(w*0.5f,h*0.7f), Offset(w*0.5f,h*0.85f), sw, StrokeCap.Round)
                                drawLine(col, Offset(w*0.32f,h*0.85f), Offset(w*0.68f,h*0.85f), sw, StrokeCap.Round)
                            }
                            Spacer(modifier = Modifier.height(12.dp))
                            Text("No requests yet", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Medium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            Spacer(modifier = Modifier.height(4.dp))
                            Text("Tap the microphone\nto make your first request", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f), lineHeight = 18.sp)
                        }
                    }
                }
            } else {
                LazyColumn(modifier = Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    items(sortedRequests) { request ->
                        RequestCard(request = request, onCancel = { onCancelRequest(request.id) }, onRate = { onRate(request.id) })
                    }
                }
            }
        }
    }
}

// ── Portrait layout: header top-left, mic below, requests fill rest ──
@Composable
private fun PortraitLayout(
    roomNumber: String, isDark: Boolean,
    settingsExpanded: Boolean, onToggleSettings: () -> Unit,
    wifiSsid: State<String?>, deviceIp: State<String?>,
    serverIp: String, serverPort: Int,
    onEditServer: () -> Unit, onCloseApp: () -> Unit,
    pendingConfirmation: PendingConfirmation?,
    onConfirmSubmit: () -> Unit, onConfirmCancel: () -> Unit,
    isRecording: Boolean, isProcessing: Boolean, audioLevel: Float, onMicClick: () -> Unit,
    lastTranscription: String, lastDepartment: String,
    sortedRequests: List<RequestItem>, onRefreshRequests: () -> Unit,
    onCancelRequest: (Int) -> Unit, onRate: (Int) -> Unit
) {
    Column(modifier = Modifier.fillMaxSize().padding(12.dp)) {
        HeaderCard(
            roomNumber = roomNumber, isDark = isDark,
            settingsExpanded = settingsExpanded, onToggleSettings = onToggleSettings,
            wifiSsid = wifiSsid, deviceIp = deviceIp,
            serverIp = serverIp, serverPort = serverPort,
            onEditServer = onEditServer, onCloseApp = onCloseApp,
            compact = false
        )
        Spacer(modifier = Modifier.height(6.dp))
        MicCard(
            pendingConfirmation = pendingConfirmation,
            onConfirmSubmit = onConfirmSubmit, onConfirmCancel = onConfirmCancel,
            isRecording = isRecording, isProcessing = isProcessing, audioLevel = audioLevel,
            onMicClick = onMicClick, lastTranscription = lastTranscription, lastDepartment = lastDepartment,
            modifier = Modifier.fillMaxWidth().weight(1f), narrowButtons = false
        )
        Spacer(modifier = Modifier.height(6.dp))
        RequestsPanel(
            sortedRequests = sortedRequests, onRefreshRequests = onRefreshRequests,
            onCancelRequest = onCancelRequest, onRate = onRate,
            modifier = Modifier.fillMaxWidth().weight(2f)
        )
    }
}

// ── Landscape layout: left=header+mic, right=requests full height ──
@Composable
private fun LandscapeLayout(
    roomNumber: String, isDark: Boolean,
    settingsExpanded: Boolean, onToggleSettings: () -> Unit,
    wifiSsid: State<String?>, deviceIp: State<String?>,
    serverIp: String, serverPort: Int,
    onEditServer: () -> Unit, onCloseApp: () -> Unit,
    pendingConfirmation: PendingConfirmation?,
    onConfirmSubmit: () -> Unit, onConfirmCancel: () -> Unit,
    isRecording: Boolean, isProcessing: Boolean, audioLevel: Float, onMicClick: () -> Unit,
    lastTranscription: String, lastDepartment: String,
    sortedRequests: List<RequestItem>, onRefreshRequests: () -> Unit,
    onCancelRequest: (Int) -> Unit, onRate: (Int) -> Unit
) {
    Row(modifier = Modifier.fillMaxSize().padding(8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Column(modifier = Modifier.weight(0.45f).fillMaxHeight(), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            HeaderCard(
                roomNumber = roomNumber, isDark = isDark,
                settingsExpanded = settingsExpanded, onToggleSettings = onToggleSettings,
                wifiSsid = wifiSsid, deviceIp = deviceIp,
                serverIp = serverIp, serverPort = serverPort,
                onEditServer = onEditServer, onCloseApp = onCloseApp,
                compact = true
            )
            MicCard(
                pendingConfirmation = pendingConfirmation,
                onConfirmSubmit = onConfirmSubmit, onConfirmCancel = onConfirmCancel,
                isRecording = isRecording, isProcessing = isProcessing, audioLevel = audioLevel,
                onMicClick = onMicClick, lastTranscription = lastTranscription, lastDepartment = lastDepartment,
                modifier = Modifier.fillMaxWidth().weight(1f), narrowButtons = true
            )
        }
        RequestsPanel(
            sortedRequests = sortedRequests, onRefreshRequests = onRefreshRequests,
            onCancelRequest = onCancelRequest, onRate = onRate,
            modifier = Modifier.weight(0.55f).fillMaxHeight()
        )
    }
}

suspend fun processVoiceRequest(
    audioRecorder: AudioRecorder,
    voskService: VoskService,
    nluService: NLUService,
    apiService: ApiService,
    roomNumber: String,
    lifecycleScope: kotlinx.coroutines.CoroutineScope,
    onRecordingStart: () -> Unit,
    onRecordingStop: () -> Unit,
    onProcessingStart: () -> Unit,
    onProcessingStop: () -> Unit,
    onStatusUpdate: (String) -> Unit,
    onTranscriptionUpdate: (String) -> Unit,
    onIntentUpdate: (String, Float) -> Unit,
    onAudioLevel: (Float) -> Unit,
    onSpeakResponse: (String) -> Unit,
    onSpeakAndWait: suspend (String) -> Unit,
    onAddRequest: (RequestItem) -> Unit,
    onRefreshRequests: () -> Unit,
    onComplete: () -> Unit,
    // Feature 3: Confirmation callback - fallback when voice confirmation is unclear
    onConfirmationNeeded: (String, String, Float) -> Unit,
    // Voice cancel: callback to cancel a request by ID
    onCancelRequest: (Int) -> Unit,
    // Active orders available for cancellation
    activeOrders: List<RequestItem>
) {
    if (ContextCompat.checkSelfPermission(audioRecorder.getContext(), Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
        onStatusUpdate("Permission denied"); onComplete(); return
    }

    try {
        onRecordingStart(); onStatusUpdate("Listening... (speak now)")

        val audioData = audioRecorder.recordWithVAD(
            silenceTimeoutMs = 1500L, maxDurationMs = 10000L,
            onStateChange = { state -> onStatusUpdate(state) },
            onAudioLevel = { level -> onAudioLevel(level) }
        )

        onAudioLevel(0f)
        onRecordingStop(); onProcessingStart(); onStatusUpdate("Processing...")

        lifecycleScope.launch {
            try {
                onStatusUpdate("Transcribing...")
                val transcription = voskService.transcribeAudio(audioData)
                onTranscriptionUpdate(transcription)

                if (transcription.isEmpty()) {
                    onStatusUpdate("No speech detected"); onProcessingStop(); onComplete(); return@launch
                }

                onStatusUpdate("Understanding...")
                val cleanedTranscription = cleanTranscription(transcription)

                // ── Phase 1: Detect cancel intent with a simple keyword check ──
                val wantsCancelPattern = Regex(
                    """(?:cancel|remove|stop|drop|delete|withdraw|don'?t\s+need|never\s*mind|forget\s+about)""",
                    RegexOption.IGNORE_CASE
                )

                if (wantsCancelPattern.containsMatchIn(cleanedTranscription)) {
                    onIntentUpdate("cancel_request", 0.99f)

                    val cancellableOrders = activeOrders.filter { it.status in listOf("pending", "in_progress") }

                    if (cancellableOrders.isEmpty()) {
                        onSpeakResponse("You have no active requests that can be cancelled.")
                        onProcessingStop(); onComplete(); return@launch
                    }

                    // ── Phase 2: Try to resolve the target order ──
                    // First attempt: resolve directly from the original utterance
                    var resolved = resolveOrderFromReply(cleanedTranscription, cancellableOrders)

                    if (resolved == null) {
                        if (cancellableOrders.size == 1) {
                            val order = cancellableOrders[0]
                            val orderSummary = "You have one active request: ${order.department}. Should I cancel it?"
                            onSpeakAndWait(orderSummary)

                            onStatusUpdate("Say Yes or No...")
                            onProcessingStop(); onRecordingStart()
                            val confirmAudio = audioRecorder.recordWithVAD(
                                silenceTimeoutMs = 2000L, maxDurationMs = 5000L,
                                onAudioLevel = { level -> onAudioLevel(level) }
                            )
                            onAudioLevel(0f); onRecordingStop(); onProcessingStart()

                            val confirmText = voskService.transcribeAudio(confirmAudio).lowercase().trim()
                            onTranscriptionUpdate(confirmText)
                            if (listOf("yes", "yeah", "yep", "sure", "ok", "okay", "go ahead", "do it", "proceed").any { confirmText.contains(it) }) {
                                resolved = order
                            } else {
                                onSpeakResponse("Okay, I'll keep your request.")
                                onProcessingStop(); onComplete(); return@launch
                            }
                        } else {
                            onSpeakAndWait("Please tell just the order number to cancel.")

                            onStatusUpdate("Say the order number now...")
                            onProcessingStop(); onRecordingStart()
                            val replyAudio = audioRecorder.recordWithVAD(
                                silenceTimeoutMs = 2000L, maxDurationMs = 6000L,
                                onAudioLevel = { level -> onAudioLevel(level) }
                            )
                            onAudioLevel(0f); onRecordingStop(); onProcessingStart()

                            val replyText = voskService.transcribeAudio(replyAudio).lowercase().trim()
                            onTranscriptionUpdate(replyText)
                            resolved = resolveOrderFromReply(replyText, cancellableOrders)

                            if (resolved == null && replyText.isNotEmpty()) {
                                onSpeakAndWait("Sorry, I didn't catch that. Please say the order number again, for example one zero five.")
                                onProcessingStop(); onRecordingStart()
                                val retryAudio = audioRecorder.recordWithVAD(
                                    silenceTimeoutMs = 2000L, maxDurationMs = 6000L,
                                    onAudioLevel = { level -> onAudioLevel(level) }
                                )
                                onAudioLevel(0f); onRecordingStop(); onProcessingStart()
                                val retryText = voskService.transcribeAudio(retryAudio).lowercase().trim()
                                onTranscriptionUpdate(retryText)
                                resolved = resolveOrderFromReply(retryText, cancellableOrders)
                            }
                        }
                    }

                    if (resolved == null) {
                        onSpeakResponse("I'm sorry, I couldn't identify which request to cancel. Please try again or use the cancel button in your request history.")
                        onProcessingStop(); onComplete(); return@launch
                    }

                    // Confirm before cancelling
                    val confirmMsg = "You want to cancel request ${resolved.id}, ${resolved.department}. Should I proceed?"
                    onSpeakAndWait(confirmMsg)

                    onStatusUpdate("Say Yes or No...")
                    onProcessingStop(); onRecordingStart()
                    val confirmAudio = audioRecorder.recordWithVAD(
                        silenceTimeoutMs = 2000L, maxDurationMs = 5000L,
                        onAudioLevel = { level -> onAudioLevel(level) }
                    )
                    onAudioLevel(0f); onRecordingStop(); onProcessingStart()

                    val confirmText = voskService.transcribeAudio(confirmAudio).lowercase().trim()
                    onStatusUpdate("You said: \"$confirmText\"")

                    val yesWords = listOf("yes", "yeah", "yep", "sure", "ok", "okay", "go ahead", "do it", "proceed")

                    when {
                        yesWords.any { confirmText.contains(it) } -> {
                            onCancelRequest(resolved.id)
                            onProcessingStop(); onComplete()
                        }
                        else -> {
                            onSpeakResponse("Okay, I'll keep your request.")
                            onProcessingStop(); onComplete()
                        }
                    }
                    return@launch
                }

                // Retry loop: give guest up to 2 more chances if not understood
                var currentTranscription = cleanedTranscription
                var intentResult = nluService.classifyIntent(currentTranscription)
                onIntentUpdate(intentResult.name, intentResult.confidence)

                val MAX_RETRIES = 2
                val MIN_CONFIDENCE = 0.60f
                var retryCount = 0

                while (intentResult.confidence < MIN_CONFIDENCE && retryCount < MAX_RETRIES) {
                    retryCount++
                    onSpeakAndWait("I couldn't quite understand that. Could you please try again?")

                    // Listen again
                    onProcessingStop()
                    onRecordingStart()
                    onStatusUpdate("Listening... (speak now)")
                    val retryAudio = audioRecorder.recordWithVAD(
                        silenceTimeoutMs = 1500L, maxDurationMs = 10000L,
                        onStateChange = { state -> onStatusUpdate(state) },
                        onAudioLevel = { level -> onAudioLevel(level) }
                    )
                    onAudioLevel(0f)
                    onRecordingStop()
                    onProcessingStart()

                    val retryTranscription = voskService.transcribeAudio(retryAudio)
                    onTranscriptionUpdate(retryTranscription)

                    if (retryTranscription.isEmpty()) {
                        // Guest stayed silent — give up gracefully
                        onStatusUpdate("No speech detected")
                        onSpeakResponse("It seems like you don't need help right now. Feel free to tap the microphone whenever you need me.")
                        onProcessingStop(); onComplete(); return@launch
                    }

                    currentTranscription = cleanTranscription(retryTranscription)
                    intentResult = nluService.classifyIntent(currentTranscription)
                    onIntentUpdate(intentResult.name, intentResult.confidence)
                }

                // After retries exhausted, still low confidence — give up
                if (intentResult.confidence < MIN_CONFIDENCE) {
                    onStatusUpdate("Request not understood")
                    onSpeakResponse("I'm sorry, I couldn't understand your request. Could you please try again with a specific hotel service request, such as requesting towels, room cleaning, or room service?")
                    onProcessingStop(); onComplete(); return@launch
                }

                // Use the latest transcription if retries occurred
                val finalTranscription = if (retryCount > 0) currentTranscription else transcription

                // Voice confirmation: TTS asks, then listens for yes/no
                val confirmMsg = "You'd like to say: $finalTranscription. Should I submit this request?"
                onSpeakAndWait(confirmMsg)

                // Listen for voice confirmation
                onStatusUpdate("Say Yes or No...")
                onProcessingStop()
                onRecordingStart()
                val confirmAudio = audioRecorder.recordWithVAD(
                    silenceTimeoutMs = 2000L, maxDurationMs = 5000L,
                    onAudioLevel = { level -> onAudioLevel(level) }
                )
                onAudioLevel(0f)
                onRecordingStop()
                onProcessingStart()

                val confirmText = voskService.transcribeAudio(confirmAudio).lowercase().trim()
                onStatusUpdate("You said: \"$confirmText\"")

                val yesWords = listOf("yes", "yeah", "yep", "sure", "ok", "okay", "confirm", "submit", "please", "go ahead", "do it", "send")
                val noWords = listOf("no", "nope", "cancel", "don't", "stop", "never mind", "nevermind")

                when {
                    yesWords.any { confirmText.contains(it) } -> {
                        onStatusUpdate("Submitting...")
                        val handler = android.os.Handler(android.os.Looper.getMainLooper())
                        apiService.submitRequest(
                            roomNumber, finalTranscription, intentResult.name,
                            onSuccess = { response ->
                                handler.post {
                                    val request = RequestItem(response.requestId, finalTranscription, intentResult.name, intentResult.confidence, "Routing...", "pending", getCurrentTime())
                                    onAddRequest(request)
                                    onSpeakResponse("Done! Your request number ${response.requestId} has been sent to the ${response.department} team.")
                                    onProcessingStop(); onComplete()
                                    onRefreshRequests()
                                }
                            },
                            onError = {
                                handler.post {
                                    onSpeakResponse("Sorry, I could not send your request due to a network issue. Please contact the help desk using the land line phone.")
                                    onProcessingStop(); onComplete()
                                }
                            }
                        )
                    }
                    noWords.any { confirmText.contains(it) } -> {
                        onSpeakResponse("Okay, I've cancelled that request.")
                        onProcessingStop(); onComplete()
                    }
                    else -> {
                        // Unclear voice response — fall back to on-screen buttons
                        onProcessingStop()
                        onConfirmationNeeded(finalTranscription, intentResult.name, intentResult.confidence)
                    }
                }

            } catch (e: Exception) { onStatusUpdate("Error occurred"); onProcessingStop(); onComplete() }
        }
    } catch (e: Exception) { onRecordingStop(); onProcessingStop(); onComplete() }
}

// ── Department Badge ──
@Composable
fun DepartmentBadge(department: String) {
    val color = getDepartmentColor(department)
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 2.dp)) {
        Box(modifier = Modifier.size(8.dp).clip(CircleShape).background(color))
        Spacer(modifier = Modifier.width(4.dp))
        Text(department, fontSize = 11.sp, fontWeight = FontWeight.Medium, color = color)
    }
}

// ── Request Card ──
@Composable
fun RequestCard(request: RequestItem, onCancel: () -> Unit, onRate: () -> Unit) {
    val borderColor = getStatusBorderColor(request.status)
    val deptColor = getDepartmentColor(request.department)

    Card(modifier = Modifier.fillMaxWidth(), elevation = CardDefaults.cardElevation(defaultElevation = 2.dp), shape = RoundedCornerShape(8.dp)) {
        Row(modifier = Modifier.fillMaxWidth().height(IntrinsicSize.Min)) {
            Box(modifier = Modifier.width(4.dp).fillMaxHeight().background(borderColor))
            Column(modifier = Modifier.fillMaxWidth().padding(10.dp)) {
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                    Text("#${request.id}", fontWeight = FontWeight.Bold, fontSize = 14.sp)
                    StatusBadge(request.status)
                }
                Spacer(modifier = Modifier.height(4.dp))
                Text(request.requestText, style = MaterialTheme.typography.bodyMedium, maxLines = 2, overflow = TextOverflow.Ellipsis)
                Spacer(modifier = Modifier.height(4.dp))
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(modifier = Modifier.size(6.dp).clip(CircleShape).background(deptColor))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(request.department, fontSize = 11.sp, color = deptColor, fontWeight = FontWeight.Medium)
                    }
                    Text(getRelativeTime(request.timestamp), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }

                // Feature 5: Cancel button for pending/in_progress requests
                // Feature 10: Rate button for completed requests + star display
                if (request.status == "pending" || request.status == "in_progress" || (request.status == "completed" && request.rating == 0)) {
                    Spacer(modifier = Modifier.height(6.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                        if (request.status == "pending" || request.status == "in_progress") {
                            OutlinedButton(
                                onClick = onCancel,
                                modifier = Modifier.height(28.dp),
                                contentPadding = PaddingValues(horizontal = 10.dp, vertical = 0.dp),
                                colors = ButtonDefaults.outlinedButtonColors(contentColor = MaterialTheme.colorScheme.error)
                            ) { Text("Cancel", fontSize = 11.sp) }
                        }
                        if (request.status == "completed" && request.rating == 0) {
                            Button(
                                onClick = onRate,
                                modifier = Modifier.height(28.dp),
                                contentPadding = PaddingValues(horizontal = 10.dp, vertical = 0.dp),
                                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.tertiary)
                            ) {
                                Icon(Icons.Default.Star, "Rate", modifier = Modifier.size(14.dp))
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("Rate Service", fontSize = 11.sp)
                            }
                        }
                    }
                }

                // Show rating stars if rated
                if (request.rating > 0) {
                    Spacer(modifier = Modifier.height(4.dp))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        for (i in 1..5) {
                            Icon(
                                Icons.Default.Star, "Star $i",
                                modifier = Modifier.size(14.dp),
                                tint = if (i <= request.rating) Color(0xFFFFB300) else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f)
                            )
                        }
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("${request.rating}/5", fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }
        }
    }
}

// ── Feature 10: Rating Dialog ──
@Composable
fun RatingDialog(requestId: Int, onRate: (Int) -> Unit, onDismiss: () -> Unit) {
    var selectedRating by remember { mutableIntStateOf(0) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Rate Service") },
        text = {
            Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.fillMaxWidth()) {
                Text("How was the service for request #$requestId?", style = MaterialTheme.typography.bodyMedium)
                Spacer(modifier = Modifier.height(16.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    for (i in 1..5) {
                        IconButton(onClick = { selectedRating = i }, modifier = Modifier.size(40.dp)) {
                            Icon(
                                Icons.Default.Star, "Star $i",
                                modifier = Modifier.size(32.dp),
                                tint = if (i <= selectedRating) Color(0xFFFFB300) else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f)
                            )
                        }
                    }
                }
                if (selectedRating > 0) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        when (selectedRating) { 1 -> "Poor"; 2 -> "Fair"; 3 -> "Good"; 4 -> "Very Good"; else -> "Excellent" },
                        fontWeight = FontWeight.Medium, color = Color(0xFFFFB300)
                    )
                }
            }
        },
        confirmButton = {
            Button(onClick = { if (selectedRating > 0) onRate(selectedRating) }, enabled = selectedRating > 0) { Text("Submit") }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("Later") }
        }
    )
}

@Composable
fun StatusBadge(status: String) {
    val isDark = isSystemInDarkTheme()
    val (backgroundColor, textColor, text) = when (status) {
        "pending" -> if (isDark) Triple(Color(0xFF4E3A00), Color(0xFFFFD54F), "Pending")
                     else Triple(Color(0xFFFFF3CD), Color(0xFF856404), "Pending")
        "in_progress" -> if (isDark) Triple(Color(0xFF0A3D47), Color(0xFF4DD0E1), "In Progress")
                         else Triple(Color(0xFFD1ECF1), Color(0xFF0C5460), "In Progress")
        "completed" -> if (isDark) Triple(Color(0xFF1B3D20), Color(0xFF81C784), "Completed")
                       else Triple(Color(0xFFD4EDDA), Color(0xFF155724), "Completed")
        "cancelled" -> if (isDark) Triple(Color(0xFF3A3A3A), Color(0xFFBDBDBD), "Cancelled")
                       else Triple(Color(0xFFE0E0E0), Color(0xFF616161), "Cancelled")
        else -> Triple(Color.Gray, Color.White, status)
    }
    Box(modifier = Modifier.background(backgroundColor, RoundedCornerShape(12.dp)).padding(horizontal = 8.dp, vertical = 2.dp)) {
        Text(text, color = textColor, fontSize = 10.sp, fontWeight = FontWeight.Bold)
    }
}

@Composable
fun ServerEditDialog(
    currentIp: String, currentPort: Int,
    onDismiss: () -> Unit, onSave: (String, Int) -> Unit
) {
    var editIp by remember { mutableStateOf(currentIp) }
    var editPort by remember { mutableStateOf(currentPort.toString()) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Server Settings") },
        text = {
            Column {
                OutlinedTextField(
                    value = editIp,
                    onValueChange = { editIp = it },
                    label = { Text("Server IP") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(modifier = Modifier.height(8.dp))
                OutlinedTextField(
                    value = editPort,
                    onValueChange = { editPort = it },
                    label = { Text("Port") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )
            }
        },
        confirmButton = {
            Button(onClick = {
                val port = editPort.toIntOrNull() ?: 8000
                onSave(editIp.trim(), port)
            }) { Text("Save") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } }
    )
}

// Return a spoken ordinal for small position numbers (1 → "first", etc.)
fun ordinalWord(n: Int): String = when (n) {
    1 -> "first"; 2 -> "second"; 3 -> "third"; 4 -> "fourth"; 5 -> "fifth"
    else -> "${n}th"
}

// Resolve which active order the user is referring to from a short reply.
// Tries: digit → word-number → ordinal → department/description keyword match.
fun resolveOrderFromReply(reply: String, activeOrders: List<RequestItem>): RequestItem? {
    val normalized = reply.lowercase().trim()

    // 1. Digit literal (e.g. "13", "order 13")
    val digit = Regex("""\d+""").find(normalized)?.value?.toIntOrNull()
    if (digit != null) return activeOrders.firstOrNull { it.id == digit }

    // 2. Spoken digit sequence like "one zero five"
    val digitSequence = spokenDigitSequenceToNumber(normalized)
    if (digitSequence != null) return activeOrders.firstOrNull { it.id == digitSequence }

    // 3. Word-to-number after stripping fillers (handles "thirteen", "the order number thirteen")
    val fromWords = wordsToNumber(normalized)
    if (fromWords != null) return activeOrders.firstOrNull { it.id == fromWords }

    // 4. Ordinals — map to position in the active orders list
    val ordinals = mapOf(
        "first" to 0, "second" to 1, "third" to 2, "fourth" to 3, "fifth" to 4,
        "last" to -1, "latest" to -1, "recent" to -1
    )
    for ((word, idx) in ordinals) {
        if (normalized.contains(word)) {
            val i = if (idx == -1) activeOrders.lastIndex else idx
            return activeOrders.getOrNull(i)
        }
    }

    // 4. Keyword match on department or request description (e.g. "the food one", "room service")
    return activeOrders.firstOrNull { order ->
        val haystack = "${order.requestText} ${order.department}".lowercase()
        normalized.split(Regex("\\s+"))
            .filter { it.length > 3 }
            .any { word -> haystack.contains(word) }
    }
}

// Filler words that are meaningless to wordsToNumber and should be ignored
private val NUMBER_FILLER_WORDS = setOf(
    "the", "my", "order", "request", "number", "booking", "no", "about", "please", "a", "it"
)

// Convert spoken number words to integer, e.g. "hundred and forty six" → 146
fun spokenDigitSequenceToNumber(text: String): Int? {
    val digitWords = mapOf(
        "zero" to 0, "one" to 1, "two" to 2, "three" to 3, "four" to 4,
        "five" to 5, "six" to 6, "seven" to 7, "eight" to 8, "nine" to 9
    )
    val words = text.lowercase().split(Regex("\\s+"))
        .filter { it.isNotBlank() }

    if (words.isEmpty()) return null
    if (words.all { digitWords.containsKey(it) }) {
        val digits = words.map { digitWords[it].toString() }.joinToString("")
        return digits.toIntOrNull()
    }
    return null
}

fun wordsToNumber(text: String): Int? {
    val ones = mapOf("zero" to 0, "one" to 1, "two" to 2, "three" to 3, "four" to 4, "five" to 5,
        "six" to 6, "seven" to 7, "eight" to 8, "nine" to 9, "ten" to 10,
        "eleven" to 11, "twelve" to 12, "thirteen" to 13, "fourteen" to 14, "fifteen" to 15,
        "sixteen" to 16, "seventeen" to 17, "eighteen" to 18, "nineteen" to 19)
    val tens = mapOf("twenty" to 20, "thirty" to 30, "forty" to 40, "fifty" to 50,
        "sixty" to 60, "seventy" to 70, "eighty" to 80, "ninety" to 90)

    val words = text.lowercase().replace("-", " ").split("\\s+".toRegex())
        .filter { it != "and" && it.isNotBlank() && it !in NUMBER_FILLER_WORDS }

    if (words.isEmpty()) return null

    // If it's already a numeric string
    words.singleOrNull()?.toIntOrNull()?.let { return it }

    var result = 0
    var current = 0

    for (word in words) {
        when {
            ones.containsKey(word) -> current += ones[word]!!
            tens.containsKey(word) -> current += tens[word]!!
            word == "hundred" || word == "hundrad" || word == "hunderd" -> current = if (current == 0) 100 else current * 100
            word == "thousand" -> { current = if (current == 0) 1000 else current * 1000; result += current; current = 0 }
            else -> return null // unrecognized word
        }
    }
    result += current
    return if (result > 0) result else null
}

fun cleanTranscription(text: String): String {
    val prefixPatterns = listOf("hi sera", "hey sera", "hello sera", "sera", "hi there", "hey there", "hello there", "hi", "hey", "hello", "excuse me", "please", "can you", "could you", "i need you to")
    var cleaned = text.lowercase().trim()
    // Strip multiple chained prefixes (e.g. "hi sera please can you cancel...")
    var changed = true
    while (changed) {
        changed = false
        for (pattern in prefixPatterns) {
            if (cleaned.startsWith(pattern)) {
                cleaned = cleaned.removePrefix(pattern).trim().removePrefix(",").removePrefix(".").trim()
                changed = true
                break
            }
        }
    }
    return cleaned.ifEmpty { text.trim() }
}

fun getCurrentTime(): String {
    val calendar = Calendar.getInstance()
    return String.format(Locale.getDefault(), "%02d:%02d", calendar.get(Calendar.HOUR_OF_DAY), calendar.get(Calendar.MINUTE))
}

fun AudioRecorder.getContext(): android.content.Context {
    val field = this.javaClass.getDeclaredField("context")
    field.isAccessible = true
    return field.get(this) as android.content.Context
}
