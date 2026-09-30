"""
The simulator's instrument universe (Guía del participante 2026, annex),
mapped to Yahoo Finance tickers.

  kind   "mx"    BMV stock, Yahoo quotes it in MXN (.MX suffix)
         "fibra" BMV FIBRA (REIT), MXN
         "sic"   US stock traded via the SIC; Yahoo quote in USD -> x USDMXN
         "etf"   US ETF via the SIC; USD -> x USDMXN
  dead   the company no longer trades under this ticker (acquired/merged).
         The simulator still lets you buy them; we never touch them.

Funds (ACTI500, JPMRVUS, ...) are left out: Yahoo coverage is spotty, they are
low-vol, and we're not trading their NAV timing.

If a Yahoo symbol is wrong, fetch.py logs it in data/fetch_log.txt. Fix it here.
"""
from __future__ import annotations

# simulator name -> Yahoo symbol
MX = {
    "AC": "AC.MX", "ACTINVR": "ACTINVRB.MX", "ALFA": "ALFAA.MX", "ALPEK": "ALPEKA.MX",
    "ALSEA": "ALSEA.MX", "AMX": "AMXB.MX", "ASUR": "ASURB.MX", "BIMBO": "BIMBOA.MX",
    "BOLSA": "BOLSAA.MX", "CEMEX": "CEMEXCPO.MX", "CHDRAUI": "CHDRAUIB.MX",
    "CUERVO": "CUERVO.MX", "ELEKTRA": "ELEKTRA.MX", "FEMSA": "FEMSAUBD.MX", "GAP": "GAPB.MX",
    "GCARSO": "GCARSOA1.MX", "GCC": "GCC.MX", "GENTERA": "GENTERA.MX",
    "GFINBUR": "GFINBURO.MX", "GFNORTE": "GFNORTEO.MX", "GMEXICO": "GMEXICOB.MX",
    "GRUMA": "GRUMAB.MX", "KIMBER": "KIMBERA.MX", "KOF": "KOFUBL.MX", "LAB": "LABB.MX",
    "LASITE": "LASITEB-1.MX", "LIVEPOL": "LIVEPOLC-1.MX", "MEGA": "MEGACPO.MX",
    "MFRISCO": "MFRISCOA-1.MX", "OMA": "OMAB.MX", "ORBIA": "ORBIA.MX",
    "PE&OLES": "PE&OLES.MX", "PINFRA": "PINFRA.MX", "Q": "Q.MX", "R": "RA.MX",
    "SITES1": "SITES1A-1.MX", "VESTA": "VESTA.MX", "VOLAR": "VOLARA.MX",
    "WALMEX": "WALMEX.MX", "BBAJIO": "BBAJIOO.MX",
}

FIBRA = {"FIBRAMQ": "FIBRAMQ12.MX", "FIBRAPL": "FIBRAPL14.MX", "FUNO": "FUNO11.MX",
         "TERRA": "TERRA13.MX"}

# SIC names; simulator suffixes like AA1/CCL1/OXY1 map to the plain US ticker
_SIC_RAW = """XYZ AA1:AA AAL AAPL ABBV ABNB AFRM AGNC AMAT AMD AMZN AVGO AXP BA BABA BAC BMY
BRKB:BRK-B C CAT CCL1:CCL CLF COST CPE CRM CSCO CVS CVX DAL DIS DVN ETSY F FANG FCX FDX
FSLR FUBO GE GM GME GOOGL HD INTC JNJ JPM KO LCID LLY LUV LVS MA MARA MCD MELI META MRK
MRNA MRO MSFT MU NCLH NFLX NKE NU NVAX NVDA ORCL OXY1:OXY PARA PEP PFE PG PINS PLTR PYPL
QCOM RCL RIOT RIVN SBUX SHOP SOFI SPCE T TGT TMO TSLA TSM TX UAL UBER UNH UPST V VZ WFC
WMT XOM ZM"""
SIC = dict(t.split(":") if ":" in t else (t, t) for t in _SIC_RAW.split())

ETF = {t: t for t in """AAXJ ACWI BIL BOTZ DIA EEM EWZ GDX GLD IAU ICLN INDA IVV KWEB LIT
MCHI PSQ QCLN QQQ SHV SHY SLV SOXX SPLG SPY TAN TLT USO VEA VGT VNQ VOO VT VTI VWO VYM
XLE XLF XLK XLV""".split()}

# acquired / merged: still listed in the simulator, don't trade them
DEAD = {"CPE", "MRO", "PARA"}

FX = "USDMXN=X"
BENCH = {"IPC": "^MXX", "SPX": "^GSPC"}


def universe() -> list[dict]:
    rows = []
    for kind, table in (("mx", MX), ("fibra", FIBRA), ("sic", SIC), ("etf", ETF)):
        for name, yf_sym in table.items():
            rows.append(dict(name=name, yahoo=yf_sym, kind=kind, usd=kind in ("sic", "etf"),
                             dead=name in DEAD))
    return rows


def live() -> list[dict]:
    return [r for r in universe() if not r["dead"]]


if __name__ == "__main__":
    u = universe()
    from collections import Counter
    print(Counter(r["kind"] for r in u), "dead:", sorted(DEAD))
