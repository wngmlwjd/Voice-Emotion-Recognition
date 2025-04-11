window.addEventListener('DOMContentLoaded', function () {
    const recordButton = document.getElementById('recordButton');
    const audioElement = document.getElementById('audioPlayback'); // 오디오 재생 요소
    const timerDisplay = document.getElementById('timerDisplay'); // 시간 표시 요소
    const uploadButton = document.getElementById('uploadButton'); // 파일 업로드 버튼
    const selectedFileName = document.getElementById('fileName'); // 선택된 파일 이름 표시
    const analysisButton = document.getElementById('analysisButton'); // 분석 버튼
    let recorder;
    let audioStream;
    let timerInterval;
    let elapsedTime = 0; // 경과 시간 초기화
    let fileName

    // 타이머를 갱신하는 함수
    function updateTimer() {
        elapsedTime++;
        const minutes = Math.floor(elapsedTime / 60);
        const seconds = elapsedTime % 60;
        timerDisplay.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }

    recordButton.addEventListener('click', async function () {
        let now, year, month, day, hours, minutes, seconds;

        if (recordButton.value === '녹음 시작') {
            // 마이크 접근 및 녹음 시작
            try {
                audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                recorder = new RecordRTC(audioStream, {
                    type: "audio",
                    mimeType: "audio/wav",
                    recorderType: StereoAudioRecorder,
                    numberOfAudioChannels: 1,
                    // sampleRate: 44100
                });

                recorder.startRecording();
                recordButton.value = '녹음 중지';

                now = new Date();
                year = now.getFullYear();
                month = String(now.getMonth() + 1).padStart(2, '0');  // 월은 0부터 시작하므로 +1
                day = String(now.getDate()).padStart(2, '0');
                hours = String(now.getHours()).padStart(2, '0');
                minutes = String(now.getMinutes()).padStart(2, '0');
                seconds = String(now.getSeconds()).padStart(2, '0');
                
                fileName = `${year}-${month}-${day}_${hours}-${minutes}-${seconds}.wav`;

                // 타이머 시작
                timerInterval = setInterval(updateTimer, 1000); // 1초마다 updateTimer 호출
            } catch (error) {
                alert("마이크 접근 오류");
                console.error("마이크 접근 오류:", error);
            }
        } else {
            // 녹음 중지 및 파일 저장
            recorder.stopRecording(() => {
                let audioBlob = recorder.getBlob();
                let audioUrl = URL.createObjectURL(audioBlob);

                // 녹음된 오디오 재생
                audioElement.src = audioUrl;
                selectedFileName.innerText = fileName

                // 마이크 스트림 해제
                audioStream.getTracks().forEach(track => track.stop());

                // 타이머 중지
                clearInterval(timerInterval);
                elapsedTime = 0; // 경과 시간 초기화

                recordButton.value = '녹음 시작';
                timerDisplay.textContent = '00:00'; // 시간 표시 초기화

                // 서버에 오디오 전송
                const formData = new FormData();
                formData.append('audio', audioBlob, fileName);

                fetch('/save_record', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    console.log('서버 응답:', data);
                })
                .catch(error => {
                    console.error('오디오 전송 오류:', error);
                })
            });
        }
    });
    
    uploadButton.addEventListener('change', function () {
        const file = uploadButton.files[0];  // 선택된 첫 번째 파일

        if (file) {
            let audioUrl = URL.createObjectURL(file);

            audioElement.src = audioUrl;
            selectedFileName.innerText = file.name;
        }
    });

    analysisButton.addEventListener('click', async function () {
        const audioSrc = audioElement.src;
        const resultDisplay = document.getElementById('resultDisplay');
        const contentDisplay = document.getElementById('contentDisplay');

        if(!audioSrc) {
            alert('오디오 파일을 녹음하거나 업로드하세요.');
            return;
        }

        if(analysisButton.value === '분석 시작') {
            analysisButton.value = '분석 중...';

            try {
                // audioPlayback에 로드된 오디오 파일을 Blob으로 가져오기
                const response = await fetch(audioSrc);
                const audioBlob = await response.blob();
                const file = uploadButton.files?.[0];

                // predict

                // FormData로 서버에 전송
                let formData = new FormData();
                if (file) {
                    // 업로드된 파일이 있는 경우
                    formData.append('audio', audioBlob, file.name);
                } else {
                    // 업로드 파일이 없으면 녹음된 파일로 처리
                    formData.append('audio', audioBlob, fileName); // 이때 fileName은 전역변수
                }

                let result = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                let data = await result.json();
                console.log('분석 결과:', data);

                const prediction = data.predictions;

                // recommend
                formData = new FormData();
                formData.append('emotion', prediction);

                result = await fetch('/recommend', {
                    method: 'POST',
                    body: formData
                });

                const movieData = await result.json();
                console.log('추천 결과:', movieData);
                
                // 결과 출력
                resultDisplay.textContent = `분석 결과: ${prediction}`;
                contentDisplay.innerHTML = `[추천 컨텐츠] <br><br>
                    --------------------------------- <br>`;
                for(let i = 0;i < 10;i++) {
                    let poster_url = 'https://image.tmdb.org/t/p/w200/' + movieData.results[i].poster_path + ".jpg";

                    contentDisplay.insertAdjacentHTML("beforeend", `
                        <span id="content${i + 1}" class="contents">
                            <img src="${poster_url}" id="poster${i + 1}" class="posters">
                            <p id="description${i + 1}" class="descriptions">
                                #${i + 1} <br><br>
                                영화 제목: ${movieData.results[i].title} <br>
                                개봉일: ${movieData.results[i].release_date} |
                                평점: ${movieData.results[i].vote_average} <br><br>
                                개요: ${movieData.results[i].overview || "없음"} <br>
                            </p>
                        </span>
                        <p>
                            --------------------------------- <br>
                        </p>
                    `);
                }

            } catch (err) {
                console.error('오디오 분석 요청 실패:', err);
                alert('분석 중 오류가 발생했습니다.');
            }

            analysisButton.value = '분석 시작';
        }
    });
});
