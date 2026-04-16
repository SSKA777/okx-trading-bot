import os
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

import okx.Trade as Trade
import okx.Account as Account
import okx.MarketData as MarketData
from aiogram import Bot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация из Secrets
API_KEY = os.environ.get("OKX_API_KEY")
SECRET_KEY = os.environ.get("OKX_SECRET_KEY")
PASSPHRASE = os.environ.get("OKX_PASSPHRASE")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

LEVERAGE = 2
POSITION_PERCENT = 10
TAKE_PROFIT = 5.0
STOP_LOSS = 2.0
MIN_SCORE = 4

SYMBOLS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "XRP-USDT-SWAP", "DOGE-USDT-SWAP"]


class OKXClient:
    def __init__(self):
        self.flag = "0"
        self.account = Account.AccountAPI(API_KEY, SECRET_KEY, PASSPHRASE, flag=self.flag)
        self.trade = Trade.TradeAPI(API_KEY, SECRET_KEY, PASSPHRASE, flag=self.flag)
        self.market = MarketData.MarketAPI(flag=self.flag)

    def get_usdt_balance(self) -> float:
        try:
            result = self.account.get_account_balance()
            if result['code'] == '0':
                for item in result['data'][0]['details']:
                    if item['ccy'] == 'USDT':
                        return float(item['availEq'])
        except Exception as e:
            logger.error(f"Balance error: {e}")
        return 0.0

    def get_candles(self, symbol: str, limit: int = 100) -> List:
        try:
            result = self.market.get_candlesticks(instId=symbol, bar="15m", limit=str(limit))
            if result['code'] == '0':
                return result['data']
        except Exception as e:
            logger.error(f"Candles error: {e}")
        return []

    def open_position(self, symbol: str, direction: str) -> Dict:
        try:
            self.account.set_leverage(instId=symbol, lever=str(LEVERAGE), mgnMode="isolated")
            ticker = self.market.get_ticker(instId=symbol)
            if ticker['code'] != '0':
                return {"success": False, "error": "No price"}
            price = float(ticker['data'][0]['last'])
            balance = self.get_usdt_balance()
            size = str(int((balance * POSITION_PERCENT / 100) * LEVERAGE / price))
            side = "buy" if direction == "long" else "sell"
            tp_price = str(round(price * (1 + TAKE_PROFIT / 100) if direction == "long" else price * (1 - TAKE_PROFIT / 100), 2))
            sl_price = str(round(price * (1 - STOP_LOSS / 100) if direction == "long" else price * (1 + STOP_LOSS / 100), 2))
            result = self.trade.place_order(
                instId=symbol, tdMode="isolated", side=side, ordType="market",
                sz=size, tpTriggerPx=tp_price, tpOrdPx=tp_price,
                slTriggerPx=sl_price, slOrdPx=sl_price
            )
            return {"success": result['code'] == '0', "symbol": symbol, "direction": direction,
                    "price": price, "size": size, "tp": tp_price, "sl": sl_price}
        except Exception as e:
            return {"success": False, "error": str(e)}


class Analyzer:
    def analyze(self, symbol: str, candles: List) -> Optional[Dict]:
        if len(candles) < 50:
            return None
        closes = np.array([float(c[4]) for c in candles])
        volumes = np.array([float(c[5]) for c in candles])
        rsi = self._rsi(closes)
        macd, signal = self._macd(closes)
        bb_upper, _, bb_lower = self._bollinger(closes)
        price = closes[-1]
        score = 0
        direction = None
        if rsi[-1] < 35:
            score += 2
            direction = "long"
        elif rsi[-1] > 65:
            score += 2
            direction = "short"
        if macd[-1] > signal[-1] and macd[-2] <= signal[-2]:
            score += 2
            direction = "long"
        elif macd[-1] < signal[-1] and macd[-2] >= signal[-2]:
            score += 2
            direction = "short"
        if price <= bb_lower[-1]:
            score += 1
            direction = "long"
        elif price >= bb_upper[-1]:
            score += 1
            direction = "short"
        if volumes[-1] > np.mean(volumes[-20:]) * 1.5:
            score += 1
        if score >= MIN_SCORE and direction:
            return {"symbol": symbol, "direction": direction, "score": score, "price": price, "rsi": rsi[-1]}
        return None

    def _rsi(self, prices, period=14):
        deltas = np.diff(prices)
        seed = deltas[:period + 1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        rsi = np.zeros_like(prices)
        rsi[:period] = 100 - 100 / (1 + rs)
        for i in range(period, len(prices)):
            delta = deltas[i - 1]
            upval, downval = (delta, 0) if delta > 0 else (0, -delta)
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 100
            rsi[i] = 100 - 100 / (1 + rs)
        return rsi

    def _macd(self, prices, fast=12, slow=26, sig=9):
        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)
        macd = ema_fast - ema_slow
        signal = self._ema(macd, sig)
        return macd, signal

    def _bollinger(self, prices, period=20, std=2):
        sma = np.convolve(prices, np.ones(period) / period, mode='same')
        std_arr = np.array([np.std(prices[max(0, i - period + 1):i + 1]) for i in range(len(prices))])
        return sma + std * std_arr, sma, sma - std * std_arr

    def _ema(self, data, period):
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema


async def main():
    logger.info(f"Start: {datetime.now()}")
    okx = OKXClient()
    analyzer = Analyzer()
    bot = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None
    balance = okx.get_usdt_balance()
    logger.info(f"Balance: ${balance:.2f}")
    signals = []
    for symbol in SYMBOLS:
        candles = okx.get_candles(symbol)
        if candles:
            signal = analyzer.analyze(symbol, candles)
            if signal:
                signals.append(signal)
                logger.info(f"Signal: {symbol} - {signal['direction']} (score: {signal['score']})")
    if signals:
        best = max(signals, key=lambda x: x['score'])
        result = okx.open_position(best['symbol'], best['direction'])
        if bot:
            if result.get('success'):
                msg = (f"✅ **СДЕЛКА**\n\n{best['symbol']}\n📈 {best['direction'].upper()}\n"
                       f"💰 ${best['price']:.4f}\n🎯 TP: {result['tp']}\n🛑 SL: {result['sl']}\n⭐ Score: {best['score']}")
            else:
                msg = f"❌ Ошибка: {result.get('error')}"
            await bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode="Markdown")
            await bot.session.close()
        logger.info(f"Trade: {result}")
    else:
        logger.info("No signals")
        if bot:
            await bot.session.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
