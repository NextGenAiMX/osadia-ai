"""
OSADIA Sovereign Dialect Recognition & Adaptation Engine (v2.0)
================================================================
Real-time Mexican Regional Spanish identification, rapid turn-based calibration,
and strict language policy enforcement (Mexican Spanish by default; English strictly upon request).

Governed by: Next GenAI Mexico S.A. de C.V. / OSADIA Sovereign Architecture
Copyright (c) 2026 Rob Kucher & Rebekah AI
"""

import re
import json
import sqlite3
import os
from typing import Dict, Any, Tuple, Optional, List

# Comprehensive Regional Dialect Signatures (7 Mexican Regions + Key LATAM Varieties)
DIALECT_SIGNATURES = {
    "mexico_norte": {
        "region_name": "México (Norte - Sonora, Sinaloa, Chihuahua, Coahuila, Nuevo León, Tamaulipas)",
        "keywords": [
            "plebe", "morro", "morra", "machín", "fierro", "troca", "cura", "curada", "arre", 
            "jalar", "bien acá", "al tiro", "chilo", "chila", "bichi", "sordearse", "guacho", 
            "huerco", "huerquilla", "chiripera", "chopo", "asina"
        ],
        "syntactic": ["al tiro", "de amadre", "bien mucho"],
        "tone": "norteno",
        "greeting": "¡Qué onda! ¿Qué tal todo por el norte?",
        "lexical_adaptations": {"genial": "chilo / con madre", "trabajar": "jalar", "amigo": "compa / plebe", "camioneta": "troca"}
    },
    "mexico_centro_cdmx": {
        "region_name": "México (Centro / Altiplano - CDMX, Edomex, Morelos)",
        "keywords": [
            "chido", "chida", "órale", "qué onda", "güey", "wey", "chale", "no manches", 
            "neta", "padrísimo", "a poco", "cámara", "carnal", "nel", "simón", "rifa", 
            "valedor", "chilangolandia", "aguanta", "de volada", "vara", "changarro"
        ],
        "syntactic": ["cámara carnal", "ni de chiste", "qué hubo"],
        "tone": "chilango_institucional",
        "greeting": "¡Qué hubo! Un gusto saludarte.",
        "lexical_adaptations": {"genial": "chido", "amigo": "carnal / valedor", "cierto": "neta", "rápido": "en corto"}
    },
    "mexico_occidente_bajio": {
        "region_name": "México (Occidente / Bajío - Jalisco, Guanajuato, Querétaro, Aguascalientes, Michoacán)",
        "keywords": [
            "edá", "ira", "ocupo", "bien mucho", "sabe", "chopear", "guanatos", "asina", 
            "ansina", "arrempujar", "ganso", "fuchi", "pégate pa'llá", "ocupamos"
        ],
        "syntactic": ["ocupo que", "bien mucho", "ira nomás"],
        "tone": "occidente_bajio",
        "greeting": "¡Qué tal! Mucho gusto en saludarte.",
        "lexical_adaptations": {"necesito": "ocupo", "mira": "ira", "bastante": "bien mucho"}
    },
    "mexico_costeno_golfo": {
        "region_name": "México (Costeño / Golfo - Veracruz, Tabasco, Guerrero)",
        "keywords": [
            "manito", "manita", "chocojic", "pachanga", "chocante", "pelaná", "peje", 
            "huachinango", "marmaja", "chispas", "coño", "mi rey", "compadrito"
        ],
        "syntactic": ["manito", "de volón", "al toque"],
        "tone": "costeno_calido",
        "greeting": "¡Qué pasó, manito! Qué gusto saludarte.",
        "lexical_adaptations": {"amigo": "manito", "fiesta": "pachanga"}
    },
    "mexico_peninsular_yucatan": {
        "region_name": "México (Peninsular - Yucatán, Campeche, Quintana Roo)",
        "keywords": [
            "chichí", "mox", "xix", "loch", "pelaná", "hacer chuchul", "masiosare", 
            "huiro", "x'la", "mulix", "tuch", "pibil", "k'iwik"
        ],
        "syntactic": ["lo busco y no lo busco", "hacer loch"],
        "tone": "peninsular_yucateco",
        "greeting": "¡Hola! Un gusto saludarte desde tierras peninsulares.",
        "lexical_adaptations": {"abuela": "chichí", "migaja / residuo": "xix", "abrazar": "hacer loch"}
    },
    "mexico_puebla_tlaxcala": {
        "region_name": "México (Puebla / Tlaxcala)",
        "keywords": [
            "pipope", "camote", "achichincle", "chito", "cocol", "molote", "chamuco", 
            "acocote", "chaneque"
        ],
        "syntactic": ["con el debido respeto", "mande usted"],
        "tone": "formal_cortes",
        "greeting": "Muy buenos días. A sus órdenes y con gusto le asisto.",
        "lexical_adaptations": {"ayudante": "achichincle", "dulce": "camote"}
    },
    # Latin American Regional Signatures
    "rioplatense": {
        "region_name": "Rioplatense (Argentina / Uruguay)",
        "keywords": ["che", "boludo", "pelotudo", "laburo", "posta", "pibe", "quilombo", "guacho", "mina", "morfar", "chabón", "macanudo", "guita"],
        "syntactic": ["tenés", "sos", "hacés", "vení", "mirá", "decime"],
        "tone": "rioplatense",
        "greeting": "¡Hola! ¿Cómo andás, che?",
        "lexical_adaptations": {"trabajo": "laburo", "cierto": "posta", "dinero": "guita"}
    },
    "andino_colombia": {
        "region_name": "Colombia (Andina / Paisa / Bogotano)",
        "keywords": ["parce", "parcero", "chévere", "bacano", "vaina", "chimba", "berraco", "gonorrea", "sumercé", "a la orden"],
        "syntactic": ["a la orden", "sumercé", "venga le digo"],
        "tone": "colombiano",
        "greeting": "¡Buenas! ¿Cómo le va, parce? Con gusto a la orden.",
        "lexical_adaptations": {"amigo": "parce", "excelente": "bacano / chévere"}
    },
    "chile": {
        "region_name": "Chile",
        "keywords": ["cachái", "cachai", "po", "weón", "wn", "la raja", "fome", "pololo", "polola", "bkn", "bacán"],
        "syntactic": ["erí", "tení", "estái", "po"],
        "tone": "chileno",
        "greeting": "¡Hola! ¿Cómo estái?",
        "lexical_adaptations": {"entiendes": "cachái", "aburrido": "fome"}
    }
}

ENGLISH_TRIGGER_PATTERNS = [
    re.compile(r"\b(in english|speak english|answer in english|respond in english|reply in english)\b", re.I),
    re.compile(r"\b(can we speak english|english only|english please|switch to english)\b", re.I),
    re.compile(r"\b(translate .* to english|explain .* in english|write .* in english)\b", re.I),
    re.compile(r"\b(speak (in )?english|talk in english)\b", re.I)
]

class OSADIADialectEngine:
    """
    Stateful Dialect Engine:
      1. Enforces Mexican Spanish by default across all contexts.
      2. Strictly restricts English output to explicit user requests.
      3. Identifies the user's regional dialect within the first 1-3 prompts.
      4. Smoothly calibrates conversational warmth, vocabulary, and phrasing.
    """
    def __init__(self, memory_db_path: str = "/home/drt/.nga-cli/memory/database/rebekah_memory.db"):
        self.memory_db_path = memory_db_path
        self._init_memory_table()

    def _init_memory_table(self):
        try:
            os.makedirs(os.path.dirname(self.memory_db_path), exist_ok=True)
            conn = sqlite3.connect(self.memory_db_path, timeout=10)
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS osadia_user_dialects (
                user_id TEXT PRIMARY KEY,
                preferred_dialect TEXT,
                region_name TEXT,
                tone TEXT,
                turn_count INTEGER DEFAULT 1,
                confidence REAL DEFAULT 0.5,
                language_preference TEXT DEFAULT 'es_MX',
                opt_out INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            conn.commit()
            conn.close()
        except Exception:
            pass

    def check_english_request(self, text: str) -> bool:
        """Determines if the user explicitly commanded English output."""
        for pattern in ENGLISH_TRIGGER_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def detect_dialect(self, text: str) -> Tuple[str, float]:
        """
        Extracts lexical and syntactic features to classify dialect origin.
        Returns: (dialect_key, confidence_score)
        """
        lower = text.lower()
        scores = {}
        for key, spec in DIALECT_SIGNATURES.items():
            match_score = 0
            for kw in spec["keywords"]:
                if re.search(r'\b' + re.escape(kw) + r'\b', lower):
                    match_score += 3
            for syn in spec.get("syntactic", []):
                if re.search(r'\b' + re.escape(syn) + r'\b', lower):
                    match_score += 4
            scores[key] = match_score

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_dialect, top_score = sorted_scores[0]

        if top_score < 3:
            # Default to Mexican Central / Standard if no specific regional markers exist
            return "mexico_centro_cdmx", 0.60

        confidence = min(0.99, 0.60 + (top_score * 0.08))
        return top_dialect, confidence

    def get_privacy_notice(self) -> str:
        """
        Returns statutory Privacy Notice compliant with Mexico's LFPDPPP (Articles 15 & 16).
        """
        return (
            "AVISO DE PRIVACIDAD SIMPLIFICADO (LFPDPPP) - OSADÍA AI / NEXT GEN AI MÉXICO:\n"
            "Los datos procesados por el motor dialectal (marcadores lingüísticos regionales e idioma de interacción) "
            "se utilizan exclusivamente para la modulación de tono y adecuación sintáctica en tiempo real, sin fines "
            "comerciales de perfilamiento discriminatorio (Art. 1 CPEUM / LFPED).\n"
            "Usted puede ejercer sus derechos ARCO (Acceso, Rectificación, Cancelación y Oposición) en cualquier "
            "momento enviando el comando '/olvidame' o '/arco-purge', lo cual eliminará inmediatamente su registro local."
        )

    def purge_user_profile(self, user_id: str) -> bool:
        """
        ARCO Right - Cancelación / Oposición:
        Permanently deletes stored user linguistic preferences from SQLite.
        """
        try:
            conn = sqlite3.connect(self.memory_db_path, timeout=10)
            cur = conn.cursor()
            cur.execute("DELETE FROM osadia_user_dialects WHERE user_id = ?", (user_id,))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        ARCO Right - Acceso:
        Retrieves the exact stored profile for the user.
        """
        try:
            conn = sqlite3.connect(self.memory_db_path, timeout=10)
            cur = conn.cursor()
            cur.execute("""
            SELECT preferred_dialect, region_name, tone, turn_count, confidence, language_preference, opt_out, updated_at
            FROM osadia_user_dialects WHERE user_id = ?
            """, (user_id,))
            row = cur.fetchone()
            conn.close()
            if row:
                return {
                    "user_id": user_id,
                    "preferred_dialect": row[0],
                    "region_name": row[1],
                    "tone": row[2],
                    "turn_count": row[3],
                    "confidence": row[4],
                    "language_preference": row[5],
                    "opt_out": bool(row[6]),
                    "updated_at": row[7]
                }
            return None
        except Exception:
            return None

    def set_user_opt_out(self, user_id: str, opt_out: bool = True) -> bool:
        """
        Opt-Out Mechanism: Disables persistent dialect profiling for this user.
        """
        try:
            conn = sqlite3.connect(self.memory_db_path, timeout=10)
            cur = conn.cursor()
            if opt_out:
                cur.execute("""
                INSERT OR REPLACE INTO osadia_user_dialects 
                (user_id, preferred_dialect, region_name, tone, turn_count, confidence, language_preference, opt_out)
                VALUES (?, 'mexico_centro_cdmx', 'México (Estándar)', 'neutral', 1, 0.5, 'es_MX', 1)
                """, (user_id,))
            else:
                cur.execute("UPDATE osadia_user_dialects SET opt_out = 0 WHERE user_id = ?", (user_id,))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def process_turn(self, user_id: str, prompt_text: str) -> Dict[str, Any]:
        """
        Evaluates a user turn:
          - Evaluates ARCO commands (/olvidame, /arco-purge, /privacidad).
          - Evaluates English permission (Strict Mexican Spanish default).
          - Tracks turns 1, 2, 3 to narrow down regional dialect.
          - Enforces LFPDPPP privacy rights.
        """
        # 0. Check ARCO Privacy Commands
        clean_text = prompt_text.strip().lower()
        if clean_text in ["/olvidame", "/forget-me", "/arco-purge", "/cancelar-perfil"]:
            self.purge_user_profile(user_id)
            return {
                "language": "es_MX",
                "dialect_key": "mexico_centro_cdmx",
                "region_name": "México (Estándar)",
                "tone": "neutral",
                "turn_count": 0,
                "confidence": 0.5,
                "arco_action": "PURGED",
                "system_prompt_addon": "CONFIRMACIÓN ARCO: El perfil de dialecto y preferencias del usuario ha sido purgado permanentemente de la base de datos conforme a la LFPDPPP."
            }

        if clean_text in ["/privacidad", "/privacy", "/aviso-privacidad"]:
            return {
                "language": "es_MX",
                "dialect_key": "mexico_centro_cdmx",
                "region_name": "México (Estándar)",
                "tone": "neutral",
                "turn_count": 0,
                "confidence": 0.5,
                "arco_action": "PRIVACY_NOTICE",
                "system_prompt_addon": self.get_privacy_notice()
            }

        # 1. Language Policy Check
        explicit_english = self.check_english_request(prompt_text)
        
        # 2. Retrieve existing user profile
        current_dialect = "mexico_centro_cdmx"
        turn_count = 1
        confidence = 0.50
        lang_pref = "en_US" if explicit_english else "es_MX"
        user_opted_out = False

        try:
            conn = sqlite3.connect(self.memory_db_path, timeout=10)
            cur = conn.cursor()
            cur.execute("""
            SELECT preferred_dialect, turn_count, confidence, language_preference, opt_out 
            FROM osadia_user_dialects WHERE user_id = ?
            """, (user_id,))
            row = cur.fetchone()
            if row:
                current_dialect = row[0]
                turn_count = row[1] + 1
                confidence = row[2]
                if not explicit_english and row[3] == "en_US":
                    lang_pref = "en_US"
                if len(row) > 4 and row[4] == 1:
                    user_opted_out = True
            conn.close()
        except Exception:
            pass

        # 3. Dialect Calibration in Turns 1 to 3 (only if not opted out)
        if not user_opted_out:
            detected_dialect, detected_conf = self.detect_dialect(prompt_text)
            if turn_count <= 3 and detected_conf > confidence:
                current_dialect = detected_dialect
                confidence = detected_conf

        spec = DIALECT_SIGNATURES.get(current_dialect, DIALECT_SIGNATURES["mexico_centro_cdmx"])

        # 4. Save state (only if not opted out)
        if not user_opted_out:
            try:
                conn = sqlite3.connect(self.memory_db_path, timeout=10)
                cur = conn.cursor()
                cur.execute("""
                INSERT OR REPLACE INTO osadia_user_dialects 
                (user_id, preferred_dialect, region_name, tone, turn_count, confidence, language_preference, opt_out)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                """, (user_id, current_dialect, spec["region_name"], spec["tone"], turn_count, confidence, lang_pref))
                conn.commit()
                conn.close()
            except Exception:
                pass

        # 5. Build System Directives
        if lang_pref == "en_US" and explicit_english:
            system_prompt_addon = (
                "The user has explicitly requested communication in English. "
                "Respond accurately and idiomatically in English, while maintaining your sovereign grounding and deep domain expertise."
            )
        else:
            system_prompt_addon = (
                f"POLÍTICA LINGÜÍSTICA ESTRICTA: Comunícate EXCLUSIVAMENTE en español mexicano. "
                f"Región identificada del interlocutor: {spec['region_name']} (Calibración: Turno {turn_count}/3, Confianza: {confidence*100:.0f}%). "
                f"Modula tu registro y calidez al tono {spec['tone']}, empleando vocabulario natural sin caer en caricaturas. "
                f"Bajo ninguna circunstancia respondas en inglés a menos que el usuario lo solicite explícitamente."
            )

        return {
            "language": lang_pref,
            "dialect_key": current_dialect,
            "region_name": spec["region_name"],
            "tone": spec["tone"],
            "turn_count": turn_count,
            "confidence": confidence,
            "system_prompt_addon": system_prompt_addon
        }

if __name__ == "__main__":
    engine = OSADIADialectEngine("/tmp/test_dialects.db")
    
    # Test 1: Default Mexican greeting
    res1 = engine.process_turn("user_123", "Hola, necesito saber cómo funciona el sistema de amparo.")
    print("Turn 1 (Neutral):", res1["region_name"], "-> Lang:", res1["language"])
    
    # Test 2: User drops northern slang in Turn 2
    res2 = engine.process_turn("user_123", "Arre compa, está con madre la explicación, pero la troca no entra en ese registro.")
    print("Turn 2 (Norteño detect):", res2["region_name"], "-> Confidence:", res2["confidence"])
    
    # Test 3: User speaks English without asking for English
    res3 = engine.process_turn("user_123", "What about the legal requirements?")
    print("Turn 3 (English without explicit switch):", res3["language"], "(Must remain es_MX)")
    
    # Test 4: User explicitly asks for English
    res4 = engine.process_turn("user_123", "Please explain the rest in English.")
    print("Turn 4 (Explicit English command):", res4["language"], "(Switches to en_US)")
