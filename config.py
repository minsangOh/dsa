"""트레이딩 봇의 설정값을 정의하는 모듈.

이 모듈은 매매 로직에 사용되는 모든 상수값을 모아 둡니다. 가격/거래대금
필터링 기준, 시스템 타이머 설정, 기술적 지표 기본값 등 각 카테고리별로
주석을 통해 설명을 제공합니다.
"""

# --- 기본 설정 ---
MIN_PRICE_KRW = 15  # 스캔 대상에 포함될 코인의 최소 가격 (원)
MIN_ORDER_VALUE_KRW = 5000  # 최소 주문 금액 (거래소 정책)
LIQUIDITY_FILTER_KRW = 1_500_000_000  # 최소 24시간 거래대금 (원)

# --- 제외 목록 ---
STABLE_COINS = ["USDT", "USDC", "DAI", "BUSD"]  # 매매에서 제외할 스테이블 코인 목록
BLOCK_LIST = ["BTC_KRW"]  # 매매에서 제외할 특정 코인 목록 (예: BTC)
MANUAL_EXCLUDE_LIST = []  # 사용자가 수동으로 추가하는 제외 목록

# --- 시스템 타이머 설정 (초) ---
SELL_CYCLE_INTERVAL_SECONDS = 3  # 매도 조건 검사 주기 (초)
MASTER_SCAN_INTERVAL_SECONDS = 300  # 매수 대상 선별, 분석, 검토 통합 주기
INDICATOR_CACHE_TTL_SECONDS = 900  # 지표 캐시 데이터 유효 시간 (초)
COOLDOWN_PERIOD_SECONDS = 1800  # 동일 심볼 재진입 대기 시간 (초)
ORDER_STALE_SECONDS = 15  # 주문이 오래되었다고 간주하고 자동 취소를 시도하는 시간 (초)

# --- 매매 설정 ---
MAX_POSITIONS = 9  # 최대 보유 포지션 수
BUY_CHUNK_DELAY_SECONDS = 1  # 매수 판단 작업 그룹 사이의 지연 시간 (초)
INITIAL_STOP_LOSS_PCT = "0.97"  # 초기 고정 손절매 비율 (%)
STOP_LOSS_ATR_MULTIPLIER = 2.0  # ATR 기반 손절매 폭 배수
POSITION_TIME_EXIT_SECONDS = (
    1200  # 매수 후 평가 기준 충족 못했을 때 강제 매도까지 허용 시간 (초)
)

# --- 트레일링 스탑 설정 ---
TRAILING_BOUNDS = [
    20,
    50,
    100,
    200,
    500,
    1000,
    2000,
    5000,
    10000,
    20000,
    50000,
    100000,
    200000,
    500000,
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
    20_000_000,
    50_000_000,
    100_000_000,
    200_000_000,
]

TRAILING_TS_PCT = [
    0.06,
    0.048,
    0.042,
    0.036,
    0.03,
    0.027,
    0.024,
    0.021,
    0.0192,
    0.018,
    0.0168,
    0.015,
    0.0138,
    0.012,
    0.0108,
    0.0096,
    0.009,
    0.0078,
    0.0066,
    0.006,
    0.0054,
    0.0048,
]

TRAILING_MIN_TICKS = [
    12,
    10,
    10,
    8,
    8,
    7,
    7,
    6,
    6,
    6,
    6,
    5,
    5,
    5,
    5,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
]

TRAILING_LIM_CUSHION = [
    8,
    7,
    6,
    5,
    5,
    4,
    4,
    4,
    4,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    2,
    2,
    2,
    2,
    2,
]

# --- 파일 관리 ---
LOG_FILES = [
    "system.log",
    "buy.log",
    "sell.log",
    "trade_fail.log",
    "errors.log",
]  # 생성 및 관리할 로그 파일 목록
CACHE_FILES = [
    "tick_sizes.cache",
]  # 프로그램 시작 시 삭제할 캐시 파일 목록

# --- 기술적 지표 기본 설정값 ---
ATR_PERIOD = 20  # ATR (평균 실제 범위) 계산 기간
RSI_PERIOD = 14  # RSI (상대 강도 지수) 계산 기간
BBANDS_PERIOD = 20  # 볼린저 밴드 계산 기간
BBANDS_STDDEV = 2.0  # 볼린저 밴드 표준편차
KELTNER_PERIOD = 20  # 켈트너 채널 계산 기간
KELTNER_ATR_MULTIPLIER = 2.0  # 켈트너 채널 ATR 배수
VOL_ZSCORE_PERIOD = 70  # 거래량 Z-점수 계산 기간
OBI_LEVELS = 20  # 오더북 불균형(OBI) 계산 시 사용할 호가 단계 수

# --- 매수 전략 1: 강력한 신고가 돌파 ---
C1_HIGH_PERIOD = 90  # 신고가 판단 기준 기간 (봉)
C1_ATR_BREAKOUT_MULTIPLIER = 1.2  # ATR을 이용한 돌파 기준 배수
C1_VOL_ZSCORE_THRESHOLD = 3.5  # 최소 거래량 Z-점수
C1_OBI_THRESHOLD = 0.6  # 최소 오더북 불균형 값 (매수 우위)

# --- 매수 전략 2: 과매도 후 반등 (Mean Reversion) ---
C2_VWAP_STDDEV_MULTIPLIER = 2.8  # VWAP 하단 밴드 표준편차 배수
C2_RSI_OVERSOLD_THRESHOLD = 18.0  # RSI 과매도 기준값
C2_VOL_ZSCORE_THRESHOLD = 1.5  # 최소 거래량 Z-점수

# --- 매수 전략 3: 저점 이탈 후 반등 (Stop Hunt Reversal) ---
C3_LOW_PERIOD = 120  # 신저가 판단 기준 기간 (봉)
C3_WICK_BODY_RATIO = 2.0  # 몸통 대비 아래 꼬리 최소 비율
C3_VOL_ZSCORE_THRESHOLD = 2.5  # 최소 거래량 Z-점수

# --- 매수 전략 4: 스퀴즈 후 변동성 폭발 (Squeeze Breakout) ---
C4_SQUEEZE_PERIOD = 40  # 스퀴즈 상태 최소 지속 기간 (봉)
C4_SQUEEZE_THRESHOLD = 0.7  # 볼린저밴드 폭 / 켈트너채널 폭 비율 기준
C4_RANGE_ATR_MULTIPLIER = 2.0  # 돌파 시 최소 캔들 범위 (ATR 배수)
C4_VOL_ZSCORE_THRESHOLD = 2.5  # 최소 거래량 Z-점수

# --- 매수 전략 5: 분봉 틱 상승 돌파 ---
C5_1MIN_TICK_RISE = 10  # 1분봉 기준 직전 봉 최소 틱 상승 횟수
C5_2MIN_TICK_RISE = 18  # 직전 2개 1분봉 누적 틱 상승 허용 상한

# C5 전략 필터: 거래량
C5_VOLUME_SHORT_PERIOD = 3  # 단기 거래량 평균 계산 기간 (5분봉 * 3 = 15분)
C5_VOLUME_LONG_PERIOD = 12  # 장기 거래량 평균 계산 기간 (5분봉 * 12 = 60분)
C5_VOLUME_SPIKE_FACTOR = (
    1.8  # 단기 평균 거래량이 장기 평균 대비 몇 배 이상이어야 하는지 지정
)

# C5 전략 필터: 오더북
C5_ORDERBOOK_TICKS = 10  # 오더북 분석에 사용할 호가 단위(틱) 수
C5_ORDERBOOK_RATIO = 1.35  # 매수세/매도세 비율 최소값

# C5 전략 필터: 매도벽
C5_SELL_LIQUIDITY_RANGE_PCT = 1.0  # 현재가 대비 매도벽을 확인할 가격 범위 (%)
C5_SELL_LIQUIDITY_FACTOR = 2.0  # 평균 대비 매도벽이 얼마나 두꺼운지 판단하는 기준 배수
