import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from services.face_analyzer import analyze_uploaded_image

app = FastAPI(title="HairPick Face-TI AI System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analyze-face")
async def api_analyze_face(
    file: UploadFile = File(...),
    name: str = Form("사용자"),
    age: str = Form(""),
    gender: str = Form("")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")
    try:
        contents = await file.read()
        report = analyze_uploaded_image(contents, user_name=name)
        if not report["success"]:
            return JSONResponse(status_code=422, content=report)
        return report
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_content = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="theme-color" content="#4f46e5">
    <title>헤어픽 (HairPick) - AI Face-TI 정밀 진단</title>
    
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
    
    <style>
        @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
        body { font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif; }
        .gradient-card { background: linear-gradient(145deg, #1e1b4b 0%, #0f172a 100%); }
        .fade-in { animation: fadeIn 0.35s ease-out forwards; }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body class="bg-slate-100 text-slate-900 min-h-screen flex justify-center selection:bg-indigo-500 selection:text-white">

    <div class="w-full max-w-md bg-white min-h-screen shadow-2xl flex flex-col relative overflow-x-hidden">

        <!-- STEP 1: 메인 홈 & 동의서 (기존 정상 구조 100% 유지) -->
        <div id="step-1" class="flex-1 flex flex-col p-6 space-y-6 fade-in">
            <div class="text-center pt-4 space-y-2">
                <div class="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-indigo-50 text-indigo-600 text-xs font-bold tracking-wide border border-indigo-100">
                    <span>✨ AI 생체 안면 비례 분석</span>
                </div>
                <h1 class="text-3xl font-extrabold text-slate-900 tracking-tight">헤어픽 <span class="text-indigo-600">HairPick</span></h1>
                <p class="text-sm text-slate-500 font-medium">단 3초 만에 찾는 나의 고유한 얼굴 골격 DNA,<br>정밀 생체 기하 <span class="text-indigo-600 font-bold">Face-TI</span> 카드 발급</p>
            </div>

            <div class="bg-slate-50 p-5 rounded-2xl border border-slate-200/80 space-y-4">
                <h3 class="text-xs font-bold text-slate-400 uppercase tracking-wider">사용자 기본 정보</h3>
                <div class="space-y-1.5">
                    <label class="text-xs font-bold text-slate-700">이름 또는 닉네임 <span class="text-indigo-600">*</span></label>
                    <input type="text" id="user-name" placeholder="결과 카드에 표기될 이름" class="w-full px-3.5 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm bg-white">
                </div>
                <div class="grid grid-cols-2 gap-3">
                    <div class="space-y-1.5">
                        <label class="text-xs font-bold text-slate-700">나이</label>
                        <input type="number" id="user-age" placeholder="예: 25" class="w-full px-3.5 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm bg-white">
                    </div>
                    <div class="space-y-1.5">
                        <label class="text-xs font-bold text-slate-700">성별</label>
                        <select id="user-gender" class="w-full px-3.5 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm bg-white">
                            <option value="unspecified">선택 안 함</option>
                            <option value="female">여성</option>
                            <option value="male">남성</option>
                        </select>
                    </div>
                </div>
            </div>

            <div class="bg-indigo-50/50 p-5 rounded-2xl border border-indigo-100 space-y-3.5">
                <h3 class="text-xs font-bold text-indigo-900 flex items-center">
                    <i class="fa-solid fa-shield-halved mr-1.5 text-indigo-600"></i>서비스 이용 동의
                </h3>
                <label class="flex items-center space-x-2.5 text-xs font-bold text-slate-900 pb-2 border-b border-indigo-100 cursor-pointer">
                    <input type="checkbox" id="agree-all" onchange="toggleAllAgreements(this)" class="w-4 h-4 text-indigo-600 rounded focus:ring-indigo-500">
                    <span>모든 필수 약관에 전체 동의합니다</span>
                </label>
                <div class="space-y-2.5 text-xs text-slate-600">
                    <label class="flex items-start space-x-2.5 cursor-pointer">
                        <input type="checkbox" id="agree-privacy" class="agree-item w-4 h-4 text-indigo-600 rounded mt-0.5 focus:ring-indigo-500" onchange="checkAgreements()">
                        <span><b class="text-indigo-600">[필수]</b> 개인정보 수집 및 이용 동의 (진단 및 통계 분석 목적)</span>
                    </label>
                    <label class="flex items-start space-x-2.5 cursor-pointer">
                        <input type="checkbox" id="agree-photo" class="agree-item w-4 h-4 text-indigo-600 rounded mt-0.5 focus:ring-indigo-500" onchange="checkAgreements()">
                        <span><b class="text-indigo-600">[필수]</b> 안면 생체 데이터 및 사진 처리 동의 (분석 즉시 휘발)</span>
                    </label>
                </div>
            </div>

            <div class="pt-2">
                <button id="btn-to-step2" onclick="goToStep2()" disabled class="w-full py-4 rounded-2xl font-bold text-white bg-slate-300 transition duration-200 flex items-center justify-center space-x-2 shadow-lg disabled:cursor-not-allowed">
                    <span>다음: 얼굴 촬영/업로드</span>
                    <i class="fa-solid fa-arrow-right"></i>
                </button>
            </div>
        </div>

        <!-- STEP 2: 촬영 / 업로드 화면 (기존 정상 구조 100% 유지) -->
        <div id="step-2" class="hidden flex-1 flex flex-col p-6 space-y-5 fade-in">
            <div class="flex items-center justify-between">
                <button onclick="goToStep1()" class="text-xs text-slate-500 hover:text-slate-800 flex items-center font-bold">
                    <i class="fa-solid fa-chevron-left mr-1"></i>이전 단계
                </button>
                <span class="text-xs font-bold text-indigo-600">STEP 2 / 3</span>
            </div>

            <div class="text-center space-y-1">
                <h2 class="text-xl font-extrabold text-slate-900">정면 얼굴 촬영 및 확인</h2>
                <p class="text-xs text-slate-500">가이드라인 타원 안에 얼굴을 맞추고 정면을 응시하세요.</p>
            </div>

            <div class="relative bg-slate-950 rounded-2xl overflow-hidden aspect-[3/4] flex items-center justify-center shadow-inner border border-slate-800">
                <video id="video-stream" autoplay playsinline class="w-full h-full object-cover"></video>
                <img id="captured-preview" class="hidden w-full h-full object-cover" />
                <div id="camera-guide" class="absolute inset-0 pointer-events-none flex items-center justify-center">
                    <div class="w-56 h-72 border-2 border-dashed border-indigo-400/80 rounded-[50%] flex items-center justify-center">
                        <div class="w-2 h-2 rounded-full bg-indigo-400/60"></div>
                    </div>
                </div>
                <input type="file" id="file-picker" accept="image/*" class="hidden" onchange="handleFileSelected(event)">
            </div>

            <div id="controls-before-capture" class="flex items-center justify-around pt-2">
                <button onclick="document.getElementById('file-picker').click()" class="w-12 h-12 rounded-full bg-slate-100 hover:bg-slate-200 text-slate-700 flex items-center justify-center transition shadow-sm" title="앨범에서 선택">
                    <i class="fa-solid fa-images text-lg"></i>
                </button>
                <button onclick="takeSnapshot()" class="w-20 h-20 rounded-full border-4 border-indigo-500 p-1 flex items-center justify-center transition active:scale-95 shadow-md">
                    <div class="w-full h-full rounded-full bg-indigo-600 flex items-center justify-center text-white">
                        <i class="fa-solid fa-camera text-2xl"></i>
                    </div>
                </button>
                <button onclick="switchCameraFacing()" class="w-12 h-12 rounded-full bg-slate-100 hover:bg-slate-200 text-slate-700 flex items-center justify-center transition shadow-sm" title="카메라 전환">
                    <i class="fa-solid fa-camera-rotate text-lg"></i>
                </button>
            </div>

            <div id="controls-after-capture" class="hidden space-y-2.5 pt-2">
                <button onclick="startAnalysis()" class="w-full py-3.5 rounded-2xl font-bold text-sm text-white bg-indigo-600 hover:bg-indigo-700 shadow-lg shadow-indigo-200 flex items-center justify-center space-x-2 transition duration-200">
                    <i class="fa-solid fa-wand-magic-sparkles"></i>
                    <span>이 사진으로 AI 정밀 진단</span>
                </button>
                <button onclick="retakePhoto()" class="w-full py-3.5 rounded-2xl font-bold text-sm text-slate-600 bg-slate-100 hover:bg-slate-200 border border-slate-200 flex items-center justify-center space-x-2 transition duration-200">
                    <i class="fa-solid fa-camera-rotate"></i>
                    <span>다시 촬영하기</span>
                </button>
            </div>
        </div>

        <!-- 로딩 오버레이 -->
        <div id="loading-view" class="hidden fixed inset-0 bg-slate-900/80 backdrop-blur-md z-50 flex flex-col items-center justify-center text-white p-6 text-center space-y-4">
            <div class="relative">
                <div class="w-16 h-16 border-4 border-indigo-500/30 border-t-indigo-400 rounded-full animate-spin"></div>
                <i class="fa-solid fa-brain absolute inset-0 flex items-center justify-center text-indigo-400 text-xl"></i>
            </div>
            <div class="space-y-1">
                <h3 class="text-lg font-bold">생체 랜드마크 & Face-TI 분석 중</h3>
                <p class="text-xs text-slate-400">3D 자세 보정 및 14개 골격 피처를 분석하고 있습니다...</p>
            </div>
        </div>

        <!-- STEP 3: 친절한 요약 결과 및 헤어 추천 화면 -->
        <div id="step-3" class="hidden flex-1 flex flex-col p-6 space-y-5 fade-in pb-12">
            
            <div class="flex items-center justify-between">
                <span class="text-xs font-bold text-indigo-600 uppercase tracking-wide">AI Biometric Result</span>
                <button onclick="resetToBeginning()" class="text-xs text-slate-500 hover:text-slate-800 font-bold flex items-center">
                    <i class="fa-solid fa-rotate-left mr-1"></i>처음으로
                </button>
            </div>

            <!-- Face-TI 카드 -->
            <div id="face-id-card" class="gradient-card text-white p-6 rounded-3xl shadow-xl border border-indigo-500/30 space-y-4 relative overflow-hidden">
                <div class="absolute -right-8 -bottom-8 w-36 h-36 bg-indigo-500/10 rounded-full blur-2xl pointer-events-none"></div>
                <div class="flex items-center justify-between border-b border-white/10 pb-3">
                    <div class="flex items-center space-x-2">
                        <i class="fa-solid fa-fingerprint text-indigo-400 text-lg"></i>
                        <span class="text-xs font-bold tracking-widest text-slate-300">HAIRPICK FACE-TI CARD</span>
                    </div>
                    <span id="res-user-name" class="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-white/10 text-indigo-100 whitespace-nowrap"></span>
                </div>
                <div class="space-y-1 text-center py-2">
                    <span id="res-faceti-code" class="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-indigo-300 via-purple-200 to-pink-300 tracking-wider"></span>
                    <h3 id="res-faceti-title" class="text-lg font-extrabold text-white"></h3>
                    <p id="res-faceti-desc" class="text-xs text-slate-300 leading-relaxed px-2"></p>
                </div>
                <div class="bg-white/5 p-3 rounded-2xl border border-white/10 flex items-center justify-between text-xs">
                    <div>
                        <span class="text-slate-400 block text-[10px]">기본 골격 베이스</span>
                        <b id="res-base-type" class="text-white"></b>
                    </div>
                    <div class="text-right">
                        <span class="text-slate-400 block text-[10px]">분석 신뢰도</span>
                        <b id="res-base-conf" class="text-indigo-400"></b>
                    </div>
                </div>
            </div>

            <div class="bg-slate-50 p-4 rounded-2xl border border-slate-200 space-y-2.5">
                <div class="flex items-center justify-between border-b border-slate-200/80 pb-2">
                    <h4 class="text-xs font-bold text-slate-800 flex items-center">
                        <i class="fa-solid fa-shield-halved text-indigo-600 mr-1.5"></i>이미지 품질 & 부위별 측정 신뢰도
                    </h4>
                    <span id="res-quality-score" class="text-xs font-bold text-emerald-600"></span>
                </div>
                <div id="res-regional-badges" class="space-y-2 text-xs"></div>
                <p id="res-quality-guide" class="text-[11px] text-slate-500 bg-white p-2.5 rounded-xl border border-slate-200/60 leading-relaxed"></p>
            </div>

            <!-- 1. 내 얼굴 골격 요약 진단 (풀어쓴 자연어) -->
            <div class="bg-slate-50 p-5 rounded-2xl border border-slate-200 space-y-2.5">
                <h4 class="text-xs font-bold text-slate-800 flex items-center">
                    <i class="fa-solid fa-face-smile text-indigo-600 mr-1.5"></i>내 얼굴 골격 요약 진단
                </h4>
                <div id="res-friendly-summary" class="text-xs text-slate-700 leading-relaxed bg-white p-4 rounded-xl border border-slate-200/60 shadow-sm space-y-1.5"></div>
            </div>

            <!-- 2. 헤어 추천 / 비추천 (성별 배지 및 처방 이유 렌더링 영역) -->
            <div class="grid grid-cols-1 gap-3">
                <div class="bg-emerald-50/60 p-4 rounded-2xl border border-emerald-200/80 space-y-2.5">
                    <div class="flex items-center justify-between">
                        <h4 class="text-xs font-bold text-emerald-900 flex items-center">
                            <i class="fa-solid fa-circle-check text-emerald-600 mr-1.5"></i>추천 헤어 스타일 처방
                        </h4>
                        <span id="res-gender-badge" class="text-[10px] font-bold px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800"></span>
                    </div>
                    <div id="res-hair-recommended" class="space-y-2 text-xs text-emerald-900"></div>
                </div>

                <div class="bg-rose-50/60 p-4 rounded-2xl border border-rose-200/80 space-y-2">
                    <h4 class="text-xs font-bold text-rose-900 flex items-center">
                        <i class="fa-solid fa-circle-xmark text-rose-600 mr-1.5"></i>피해야 할 헤어 스타일
                    </h4>
                    <div id="res-hair-avoid" class="space-y-1.5 text-xs text-rose-900"></div>
                </div>
            </div>

            <!-- 3. 상세 분석 토글 -->
            <button onclick="toggleDetailView()" class="w-full py-3 rounded-2xl font-bold text-xs text-indigo-600 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 transition flex items-center justify-center space-x-2">
                <i id="detail-toggle-icon" class="fa-solid fa-chevron-down"></i>
                <span id="detail-toggle-text">생체 기하 분석 자세히 보기 (품질 및 전수 수치)</span>
            </button>

            <!-- 4. 토글형 상세 섹션 -->
            <div id="detail-section" class="hidden space-y-4 pt-2 border-t border-slate-200">
                <div class="bg-slate-50 p-4 rounded-2xl border border-slate-200 space-y-3">
                    <h4 id="res-primary-section-title" class="text-xs font-bold text-slate-800 flex items-center">
                        <i class="fa-solid fa-chart-simple text-indigo-600 mr-1.5"></i>1순위 메인 골격 전수 분석
                    </h4>
                    <div id="res-primary-items" class="space-y-3"></div>
                </div>

                <div class="bg-slate-50 p-4 rounded-2xl border border-slate-200 space-y-3">
                    <h4 class="text-xs font-bold text-purple-900 flex items-center">
                        <i class="fa-solid fa-layer-group text-purple-600 mr-1.5"></i>2·3순위 보조 골격 (상위 50% 이상)
                    </h4>
                    <div id="res-secondary-items" class="space-y-3"></div>
                </div>
            </div>

            <!-- 하단 버튼 -->
            <div class="space-y-2.5 pt-2">
                <button onclick="downloadCardImage()" class="w-full py-3.5 rounded-2xl font-bold text-sm text-white bg-indigo-600 hover:bg-indigo-700 shadow-lg shadow-indigo-200 flex items-center justify-center space-x-2 transition duration-200">
                    <i class="fa-solid fa-download"></i>
                    <span>Face-TI 카드 이미지 저장</span>
                </button>
                <button onclick="resetToBeginning()" class="w-full py-3.5 rounded-2xl font-bold text-sm text-slate-600 bg-slate-100 hover:bg-slate-200 border border-slate-200 flex items-center justify-center space-x-2 transition duration-200">
                    <i class="fa-solid fa-rotate-left"></i>
                    <span>새로운 사진으로 다시 검사하기</span>
                </button>
            </div>
        </div>

    </div>

    <script>
        let currentStream = null;
        let currentFacingMode = "user";
        let capturedBlob = null;

        function setText(id, text) {
            const el = document.getElementById(id);
            if (el) el.innerText = text;
        }
        function setHtml(id, html) {
            const el = document.getElementById(id);
            if (el) el.innerHTML = html;
        }

        // --- 사용자 원본 약관 로직 100% 유지 ---
        function toggleAllAgreements(master) {
            document.querySelectorAll('.agree-item').forEach(cb => cb.checked = master.checked);
            checkAgreements();
        }

        function checkAgreements() {
            const p = document.getElementById('agree-privacy')?.checked;
            const ph = document.getElementById('agree-photo')?.checked;
            const name = document.getElementById('user-name')?.value.trim() || "";
            const btn = document.getElementById('btn-to-step2');
            if (!btn) return;
            if (p && ph && name.length > 0) {
                btn.disabled = false;
                btn.className = "w-full py-4 rounded-2xl font-bold text-white bg-indigo-600 hover:bg-indigo-700 transition duration-200 flex items-center justify-center space-x-2 shadow-lg cursor-pointer";
            } else {
                btn.disabled = true;
                btn.className = "w-full py-4 rounded-2xl font-bold text-white bg-slate-300 transition duration-200 flex items-center justify-center space-x-2 shadow-lg disabled:cursor-not-allowed";
            }
        }
        document.getElementById('user-name')?.addEventListener('input', checkAgreements);

        function goToStep1() {
            document.getElementById('step-2')?.classList.add('hidden');
            document.getElementById('step-3')?.classList.add('hidden');
            document.getElementById('step-1')?.classList.remove('hidden');
            stopCamera();
        }

        function goToStep2() {
            document.getElementById('step-1')?.classList.add('hidden');
            document.getElementById('step-3')?.classList.add('hidden');
            document.getElementById('step-2')?.classList.remove('hidden');
            startCamera();
        }

        function resetToBeginning() {
            capturedBlob = null;
            document.getElementById('captured-preview')?.classList.add('hidden');
            document.getElementById('video-stream')?.classList.remove('hidden');
            document.getElementById('camera-guide')?.classList.remove('hidden');
            document.getElementById('controls-after-capture')?.classList.add('hidden');
            document.getElementById('controls-before-capture')?.classList.remove('hidden');
            goToStep1();
        }

        async function startCamera() {
            try {
                stopCamera();
                currentStream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: currentFacingMode, width: { ideal: 720 }, height: { ideal: 960 } }
                });
                const video = document.getElementById('video-stream');
                if (video) video.srcObject = currentStream;
            } catch (err) { console.warn("웹캠 접근 실패:", err); }
        }

        function stopCamera() {
            if (currentStream) { currentStream.getTracks().forEach(t => t.stop()); currentStream = null; }
        }

        function switchCameraFacing() {
            currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
            startCamera();
        }

        function takeSnapshot() {
            const video = document.getElementById('video-stream');
            if (!video) return;
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 853;
            canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

            const preview = document.getElementById('captured-preview');
            if (preview) {
                preview.src = canvas.toDataURL('image/jpeg', 0.95);
                preview.classList.remove('hidden');
            }
            video.classList.add('hidden');
            document.getElementById('camera-guide')?.classList.add('hidden');
            document.getElementById('controls-before-capture')?.classList.add('hidden');
            document.getElementById('controls-after-capture')?.classList.remove('hidden');
            canvas.toBlob(blob => { capturedBlob = blob; }, 'image/jpeg', 0.95);
        }

        function retakePhoto() {
            capturedBlob = null;
            document.getElementById('captured-preview')?.classList.add('hidden');
            document.getElementById('video-stream')?.classList.remove('hidden');
            document.getElementById('camera-guide')?.classList.remove('hidden');
            document.getElementById('controls-after-capture')?.classList.add('hidden');
            document.getElementById('controls-before-capture')?.classList.remove('hidden');
            startCamera();
        }

        function handleFileSelected(event) {
            const file = event.target.files[0];
            if (!file) return;
            capturedBlob = file;
            const reader = new FileReader();
            reader.onload = e => {
                const preview = document.getElementById('captured-preview');
                if (preview) {
                    preview.src = e.target.result;
                    preview.classList.remove('hidden');
                }
                document.getElementById('video-stream')?.classList.add('hidden');
                document.getElementById('camera-guide')?.classList.add('hidden');
                document.getElementById('controls-before-capture')?.classList.add('hidden');
                document.getElementById('controls-after-capture')?.classList.remove('hidden');
            };
            reader.readAsDataURL(file);
        }

        function toggleDetailView() {
            const detail = document.getElementById('detail-section');
            const icon = document.getElementById('detail-toggle-icon');
            const text = document.getElementById('detail-toggle-text');
            if (!detail) return;
            if (detail.classList.contains('hidden')) {
                detail.classList.remove('hidden');
                if (icon) icon.className = "fa-solid fa-chevron-up";
                if (text) text.innerText = "상세 분석 접기";
            } else {
                detail.classList.add('hidden');
                if (icon) icon.className = "fa-solid fa-chevron-down";
                if (text) text.innerText = "생체 기하 분석 자세히 보기 (품질 및 전수 수치)";
            }
        }

        async function startAnalysis() {
            if (!capturedBlob) return;
            document.getElementById('loading-view')?.classList.remove('hidden');

            const name = document.getElementById('user-name')?.value.trim() || "사용자";
            const age = document.getElementById('user-age')?.value.trim() || "";
            const gender = document.getElementById('user-gender')?.value || "";

            const formData = new FormData();
            formData.append('file', capturedBlob, 'face.jpg');
            formData.append('name', name);
            formData.append('age', age);
            formData.append('gender', gender);

            try {
                const res = await fetch('/api/analyze-face', { method: 'POST', body: formData });
                const data = await res.json();
                document.getElementById('loading-view')?.classList.add('hidden');

                if (!res.ok || !data.success) {
                    alert(data.error || "얼굴 감지 실패! 가이드라인에 맞추어 다시 촬영해 주세요.");
                    retakePhoto();
                    return;
                }
                renderResultStep(data);
            } catch (err) {
                document.getElementById('loading-view')?.classList.add('hidden');
                alert("결과 처리 중 오류가 발생했습니다: " + err.message);
            }
        }

        // --- 성별 맞춤 헤어 디자인 처방 및 기하학적 보완 원리 DB ---
        const HAIR_PRESCRIPTION_DB = {
            "HEART": {
                "female": {
                    rec: [
                        { name: "미디엄 S컬 빌드펌", reason: "턱선 아래(하안부) 지점에 A라인 웨이브 볼륨을 형성하여 뾰족한 턱 끝을 시각적으로 넓혀주고 안정감을 부여합니다." },
                        { name: "사이드뱅 레이어드컷", reason: "광대 윗선과 관자놀이를 부드러운 곡선으로 가려 넓은 상안부 폭을 슬림하게 축소합니다." },
                        { name: "소프트 풀뱅 + 태슬컷", reason: "이마 면적을 덮어 상안부 비중을 줄이고, 턱선 끝단에 수평 질감을 주어 하관 축소감을 보완합니다." }
                    ],
                    avoid: [
                        { name: "무거운 일자 풀뱅", reason: "이마를 빽빽하게 덮어 시선이 하관으로 집중되면서 턱이 더 뾰족해 보일 수 있습니다." },
                        { name: "턱 끝 볼륨이 없는 5:5 롱 생머리", reason: "수직 직선이 턱선으로 모여 하관의 좁아짐을 과도하게 강조합니다." }
                    ]
                },
                "male": {
                    rec: [
                        { name: "시스루 댄디컷", reason: "이마 양옆 여백을 가려 역삼각형 상안부 폭을 좁히고 시선을 눈매로 유도합니다." },
                        { name: "세미 리프컷 (Leaf Cut)", reason: "귀 뒤와 턱선 주변으로 떨어지는 기장감이 슬림한 턱 끝 주변의 빈 공간을 채워 골격 밸런스를 맞춥니다." },
                        { name: "소프트 애즈펌 (6:4)", reason: "이마를 좁게 오픈하여 뾰족한 턱선과의 대비를 줄이고, 관자놀이 볼륨을 다운시켜 슬림한 상안부를 연출합니다." }
                    ],
                    avoid: [
                        { name: "초단발 숏 크롭컷", reason: "상안부 가로폭만 부각되고 턱 끝 여백이 드러나 역삼각 실루엣이 도드라집니다." },
                        { name: "옆머리를 팽창시키는 하이 투블럭", reason: "옆머리가 뜨면 상안부가 더 넓어 보여 하관이 왜소해 보입니다." }
                    ]
                }
            },
            "LONG": {
                "female": {
                    rec: [
                        { name: "사이드뱅 미디엄 S컬펌", reason: "뺨 옆선에 풍성한 가로 볼륨을 형성하여 긴 세로 길이감을 시각적으로 분산시킵니다." },
                        { name: "시스루 뱅 + 레이어드컷", reason: "이마 상단 여백을 커버하여 전체 얼굴의 수직 종횡비를 아담하게 줄여줍니다." },
                        { name: "어깨선 윈드컷 / 중단발", reason: "시선이 턱 아래로 길게 늘어지지 않도록 목선에서 깔끔하게 커트하여 균형을 맞춥니다." }
                    ],
                    avoid: [
                        { name: "정수리 볼륨이 없는 5:5 롱 스트레이트", reason: "얼굴의 세로선을 그대로 노출시켜 얼굴이 더 길어 보입니다." },
                        { name: "하이 포니테일", reason: "정수리 상단 높이를 추가하여 상하 길이를 더욱 연장시킵니다." }
                    ]
                },
                "male": {
                    rec: [
                        { name: "시스루 쉐도우펌", reason: "이마를 덮어 세로 길이를 축소하고 불규칙한 컬감으로 가로 볼륨을 채워줍니다." },
                        { name: "6:4 애즈펌 (소프트 컬)", reason: "이마 노출을 최소화하면서 양옆으로 흐르는 컬을 주어 부드럽고 균형 잡힌 인상을 줍니다." },
                        { name: "댄디컷 + 옆머리 다운펌", reason: "윗머리 볼륨을 차분하게 낮추고 앞머리로 세로 라인을 끊어줍니다." }
                    ],
                    avoid: [
                        { name: "아이비리그컷 / 리젠트컷", reason: "이마 전체를 드러내고 윗머리를 세우면 얼굴 세로 길이가 극대화됩니다." },
                        { name: "5:5 가르마 장발", reason: "얼굴 중심 세로 분할선을 만들어 긴 중안부를 강조합니다." }
                    ]
                }
            },
            "SQUARE": {
                "female": {
                    rec: [
                        { name: "소프트 C컬 레이어드", reason: "턱선을 부드럽게 감싸는 곡선형 레이어로 각진 하악각 모서리를 자연스럽게 완충합니다." },
                        { name: "7:3 사이드 파팅 웨이브", reason: "사선 가르마와 웨이브 텍스처로 시선을 분산시켜 턱선 엣지를 부드럽게 연출합니다." },
                        { name: "미디엄 허쉬컷", reason: "가벼운 질감의 레이어가 하관 지지면의 묵직한 무게감을 덜어줍니다." }
                    ],
                    avoid: [
                        { name: "턱선에서 끊어지는 칼단발", reason: "하악각의 가로 너비와 수평 라인이 겹쳐 사각 턱선을 더욱 도드라지게 합니다." },
                        { name: "타이트한 올백 포니테일", reason: "얼굴 골격 외곽선 전체를 노출하여 각진 실루엣이 강조됩니다." }
                    ]
                },
                "male": {
                    rec: [
                        { name: "가일컷 (한쪽 넘김)", reason: "한쪽은 넘기고 한쪽은 떨어뜨리는 비대칭 사선 라인으로 턱선의 남성미를 세련되게 승화합니다." },
                        { name: "플랫 드롭컷", reason: "윗머리는 플랫하게 누르고 앞머리 양옆을 떨어뜨려 스퀘어 골격의 장점을 살립니다." },
                        { name: "소프트 시스루 댄디컷", reason: "부드러운 앞머리 텍스처로 강한 하관 모서리를 부드럽게 중화합니다." }
                    ],
                    avoid: [
                        { name: "무거운 바가지 투블럭", reason: "하관의 각진 너비와 상단 바가지 라인이 대비되어 사각형 윤곽이 강조됩니다." },
                        { name: "양옆이 지나치게 짧은 모히칸", reason: "하악각의 가로폭이 상대적으로 넓어 보입니다." }
                    ]
                }
            },
            "ROUND": {
                "female": {
                    rec: [
                        { name: "롱 레이어드 S컬펌", reason: "수직 방향의 긴 흐름을 만들어 동그란 볼선 실루엣을 슬림하게 교정합니다." },
                        { name: "사이드 롱 뱅 + 슬릭컷", reason: "뺨 옆선 여백을 수직으로 가려 얼굴 가로폭을 시각적으로 축소합니다." },
                        { name: "정수리 볼륨 리프컷", reason: "윗볼륨을 살려 상하 종횡비를 길어 보이게 만들어 동안 매력을 극대화합니다." }
                    ],
                    avoid: [
                        { name: "턱선 길이의 일자 둥근 단발", reason: "얼굴의 둥근 곡선과 헤어 끝선이 겹쳐 볼살이 부각됩니다." },
                        { name: "가로 볼륨만 띄우는 히피펌", reason: "얼굴 가로 너비를 확장하여 둥근 느낌을 가중시킵니다." }
                    ]
                },
                "male": {
                    rec: [
                        { name: "아이비리그컷 / 리젠트컷", reason: "이마 중앙을 시원하게 오픈하여 세로 방향의 직선감을 부여합니다." },
                        { name: "드롭컷 / 가일컷", reason: "이마를 드러내고 옆머리를 밀착시켜 둥근 턱선에 샤프한 모던 엣지를 더합니다." },
                        { name: "세미 리프 가르마펌", reason: "정수리 높이를 살리고 자연스러운 가르마로 둥근 실루엣을 타원형으로 교정합니다." }
                    ],
                    avoid: [
                        { name: "이마 전체를 둥글게 덮는 풀뱅 머리", reason: "얼굴 세로를 차단하여 가로로 넓고 둥글어 보이게 만듭니다." },
                        { name: "옆머리가 뜨는 볼륨 펌", reason: "가로 실루엣이 팽창하여 얼굴이 커 보일 수 있습니다." }
                    ]
                }
            },
            "OVAL": {
                "female": {
                    rec: [
                        { name: "태슬컷 / 슬릭펌", reason: "이상적인 계란형 윤곽을 가장 도회적이고 깔끔하게 돋보이게 합니다." },
                        { name: "글램 웨이브펌", reason: "어느 각도에서도 완벽한 황금 비례와 조화를 이루는 프리미엄 스타일입니다." },
                        { name: "올백 하이 포니테일", reason: "치우침 없는 안면 윤곽선을 시원하게 드러내어 세련미를 완성합니다." }
                    ],
                    avoid: [
                        { name: "특별한 비추천 스타일 없음", reason: "황금 비율을 지니고 있어 대부분의 기장과 스타일을 자유롭게 소화합니다." }
                    ]
                },
                "male": {
                    rec: [
                        { name: "내추럴 리프컷 / 장발 가르마", reason: "완벽한 골격 밸런스를 살려 자연스럽고 분위기 있는 무드를 연출합니다." },
                        { name: "가일컷 / 슬릭 댄디컷", reason: "단정한 이목구비와 페이스라인을 가장 돋보이게 하는 스타일링입니다." },
                        { name: "클래식 포마드 스타일", reason: "이마와 턱선의 균형을 클래식하고 스마트하게 표현합니다." }
                    ],
                    avoid: [
                        { name: "특별한 비추천 스타일 없음", reason: "어떤 커트와 펌도 조화롭게 매칭되므로 취향에 맞춰 선택하시면 됩니다." }
                    ]
                }
            }
        };

        function renderResultStep(data) {
            document.getElementById('step-2')?.classList.add('hidden');
            document.getElementById('step-3')?.classList.remove('hidden');
            stopCamera();

            // Face-TI 카드
            setText('res-user-name', `${data.user_name}님의 Face-TI`);
            setText('res-faceti-code', data.face_ti.code);
            setText('res-faceti-title', data.face_ti.title);
            setText('res-faceti-desc', data.face_ti.desc);
            setText('res-base-type', `${data.classification.rank1.label_ko}`);
            setText('res-base-conf', `${data.classification.rank1.score}%`);

            // 친절한 자연어 요약 문구 렌더링
            setHtml('res-friendly-summary', data.friendly_summary);

            // 성별(Step 1 선택값)에 맞춘 헤어 처방 매핑
            const baseTypeKey = (data.classification.rank1.type || "OVAL").toUpperCase();
            const userGender = document.getElementById('user-gender')?.value === 'male' ? 'male' : 'female';
            setText('res-gender-badge', userGender === 'male' ? '남성 맞춤 처방' : '여성 맞춤 처방');

            const shapeData = HAIR_PRESCRIPTION_DB[baseTypeKey] || HAIR_PRESCRIPTION_DB["OVAL"];
            const genderData = shapeData[userGender] || shapeData["female"];

            // 추천 헤어 렌더링 (처방 이유 포함)
            setHtml('res-hair-recommended', genderData.rec.map((item, idx) => `
                <div class="bg-white p-3 rounded-xl border border-emerald-200/80 shadow-xs space-y-1">
                    <div class="font-extrabold text-emerald-950 flex items-center">
                        <span class="w-4 h-4 rounded-full bg-emerald-600 text-white text-[10px] flex items-center justify-center mr-1.5 shrink-0">${idx+1}</span>
                        <span>${item.name}</span>
                    </div>
                    <p class="text-[11px] text-slate-600 pl-5.5 leading-relaxed"><b class="text-emerald-700">💡 처방 이유:</b> ${item.reason}</p>
                </div>
            `).join(''));

            // 비추천 헤어 렌더링
            setHtml('res-hair-avoid', genderData.avoid.map(item => `
                <div class="bg-white p-2.5 rounded-xl border border-rose-200/80 shadow-xs space-y-0.5">
                    <div class="font-bold text-rose-950 flex items-center">
                        <i class="fa-solid fa-ban text-rose-500 text-[11px] mr-1.5"></i>
                        <span>${item.name}</span>
                    </div>
                    <p class="text-[11px] text-slate-500 pl-4.5 leading-relaxed">${item.reason}</p>
                </div>
            `).join(''));

            // 상세 분석 섹션
            const q = data.quality;
            setText('res-quality-score', `종합 품질 ${q.overall_score}점 [${q.overall_grade}]`);
            setText('res-quality-guide', q.overall_guide);
            setHtml('res-regional-badges', q.regional_status.map(r => {
                const isPass = r.status === 'PASS';
                const colorClass = isPass ? 'bg-emerald-50 text-emerald-700 border-emerald-200' : 'bg-amber-50 text-amber-800 border-amber-200';
                return `<div class="p-2 rounded-lg border ${colorClass} text-[11px] font-bold"><span>${r.region} : ${r.status}</span><p class="text-[10px] font-normal opacity-90">${r.detail}</p></div>`;
            }).join(''));

            setHtml('res-primary-section-title', `<i class="fa-solid fa-chart-simple text-indigo-600 mr-1.5"></i>1순위 메인 골격 전수 분석`);
            setHtml('res-primary-items', data.primary_analysis.map((m, idx) => `
                <div class="bg-white p-3 rounded-xl border border-slate-200 space-y-1 text-xs">
                    <div class="flex justify-between font-bold"><span>${idx+1}. ${m.name}</span><span class="text-indigo-600">${m.grade}</span></div>
                    <div class="text-[11px] text-slate-500">측정값: <b>${m.value}${m.unit}</b> (평균 대비 ${m.diff_str}${m.unit})</div>
                    <p class="text-[11px] text-slate-600"><b>💡 측정 원리:</b> ${m.metric_meaning}</p>
                    ${!m.is_below_avg ? `<p class="text-[11px] text-indigo-950 bg-indigo-50/50 p-1.5 rounded"><b>👤 외모 특징:</b> ${m.visual_meaning}</p>` : ''}
                </div>`).join(''));

            if (data.secondary_traits && data.secondary_traits.length > 0) {
                setHtml('res-secondary-items', data.secondary_traits.map((m, idx) => `
                    <div class="bg-white p-3 rounded-xl border border-purple-200 space-y-1 text-xs">
                        <div class="flex justify-between font-bold"><span>${idx+1}. ${m.name}</span><span class="text-purple-600">${m.grade}</span></div>
                        <div class="text-[11px] text-slate-500">측정값: <b>${m.value}${m.unit}</b> (평균 대비 ${m.diff_str}${m.unit})</div>
                        <p class="text-[11px] text-slate-600"><b>💡 측정 원리:</b> ${m.metric_meaning}</p>
                        <p class="text-[11px] text-purple-950 bg-purple-50/50 p-1.5 rounded"><b>👤 외모 특징:</b> ${m.visual_meaning}</p>
                    </div>`).join(''));
            } else {
                setHtml('res-secondary-items', `<p class="text-center text-xs text-slate-400 py-2">2·3순위 보조 특징 없음 (단일 골격)</p>`);
            }
        }

        function downloadCardImage() {
            const card = document.getElementById('face-id-card');
            if (!card) return;
            html2canvas(card, { scale: 2, useCORS: true, backgroundColor: null }).then(canvas => {
                const link = document.createElement('a');
                link.download = `HairPick_FaceTI_${Date.now()}.png`;
                link.href = canvas.toDataURL('image/png');
                link.click();
            });
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
