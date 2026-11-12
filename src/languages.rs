//! The NLLB-200 language table.
//!
//! Exactly the 202 languages NLLB-200 has a language token for, one variant
//! each. Every listed language is supported by translation and by SONAR
//! (which shares NLLB's vocabulary); there are no placeholder entries. The
//! enum, NLLB/FLORES-200 codes, ISO codes, names and accepted aliases all
//! come from the single `languages!` table below, so they cannot drift apart.
//!
//! Synonymous names map to one variant: Filipino → `Tagalog`, Setswana →
//! `Tswana`, Nyanja → `Chichewa`, Rundi → `Kirundi`, Sepedi →
//! `NorthernSotho`. The old variant names still deserialize via serde
//! aliases.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Per-language metadata.
struct LanguageInfo {
    lang: NllbLanguage,
    /// NLLB/FLORES-200 code (ISO 639-3 + ISO 15924 script), e.g. `fra_Latn`.
    nllb: &'static str,
    /// Primary short code: ISO 639-1 where one exists, else ISO 639-3, with
    /// a `-Script` suffix for a second script of the same language.
    iso: &'static str,
    name: &'static str,
    /// Further accepted codes (ISO 639-2/639-3, macrolanguage members).
    aliases: &'static [&'static str],
}

macro_rules! languages {
    ($( $(#[$attr:meta])* $variant:ident => ($nllb:literal, $iso:literal, $name:literal, [$($alias:literal),*]) ),* $(,)?) => {
        /// A language supported by NLLB-200 (and SONAR).
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum NllbLanguage {
            $( $(#[$attr])* $variant, )*
        }

        /// Indexed by discriminant: `LANGUAGES[lang as usize].lang == lang`.
        const LANGUAGES: &[LanguageInfo] = &[
            $( LanguageInfo { lang: NllbLanguage::$variant, nllb: $nllb, iso: $iso, name: $name, aliases: &[$($alias),*] }, )*
        ];
    };
}

languages! {
    English               => ("eng_Latn", "en", "English", ["eng"]),
    Spanish               => ("spa_Latn", "es", "Spanish", ["spa"]),
    French                => ("fra_Latn", "fr", "French", ["fra"]),
    German                => ("deu_Latn", "de", "German", ["deu"]),
    Italian               => ("ita_Latn", "it", "Italian", ["ita"]),
    Portuguese            => ("por_Latn", "pt", "Portuguese", ["por"]),
    Russian               => ("rus_Cyrl", "ru", "Russian", ["rus"]),
    Polish                => ("pol_Latn", "pl", "Polish", ["pol"]),
    Silesian              => ("szl_Latn", "szl", "Silesian", []),
    Ukrainian             => ("ukr_Cyrl", "uk", "Ukrainian", ["ukr"]),
    Belarusian            => ("bel_Cyrl", "be", "Belarusian", ["bel"]),
    Czech                 => ("ces_Latn", "cs", "Czech", ["ces"]),
    Bulgarian             => ("bul_Cyrl", "bg", "Bulgarian", ["bul"]),
    Croatian              => ("hrv_Latn", "hr", "Croatian", ["hrv"]),
    Serbian               => ("srp_Cyrl", "sr", "Serbian", ["srp"]),
    Slovenian             => ("slv_Latn", "sl", "Slovenian", ["slv"]),
    Slovak                => ("slk_Latn", "sk", "Slovak", ["slk"]),
    Macedonian            => ("mkd_Cyrl", "mk", "Macedonian", ["mkd"]),
    Bosnian               => ("bos_Latn", "bs", "Bosnian", ["bos"]),
    Albanian              => ("als_Latn", "sq", "Albanian", ["alb", "sqi", "als"]),
    Greek                 => ("ell_Grek", "el", "Greek", ["ell"]),
    Romanian              => ("ron_Latn", "ro", "Romanian", ["ron"]),
    Lithuanian            => ("lit_Latn", "lt", "Lithuanian", ["lit"]),
    Latvian               => ("lvs_Latn", "lv", "Latvian", ["lav", "lvs"]),
    Latgalian             => ("ltg_Latn", "ltg", "Latgalian", []),
    Estonian              => ("est_Latn", "et", "Estonian", ["est"]),
    Irish                 => ("gle_Latn", "ga", "Irish", ["gle"]),
    Welsh                 => ("cym_Latn", "cy", "Welsh", ["cym", "wel"]),
    Scottish              => ("gla_Latn", "gd", "Scottish Gaelic", ["gla"]),
    Icelandic             => ("isl_Latn", "is", "Icelandic", ["isl", "ice"]),
    Faroese               => ("fao_Latn", "fo", "Faroese", ["fao"]),
    Norwegian             => ("nob_Latn", "no", "Norwegian (Bokmål)", ["nor", "nb", "nob"]),
    NorwegianNynorsk      => ("nno_Latn", "nn", "Norwegian (Nynorsk)", ["nno"]),
    Swedish               => ("swe_Latn", "sv", "Swedish", ["swe"]),
    Danish                => ("dan_Latn", "da", "Danish", ["dan"]),
    Dutch                 => ("nld_Latn", "nl", "Dutch", ["nld"]),
    Afrikaans             => ("afr_Latn", "af", "Afrikaans", ["afr"]),
    Luxembourgish         => ("ltz_Latn", "lb", "Luxembourgish", ["ltz"]),
    Limburgish            => ("lim_Latn", "li", "Limburgish", ["lim"]),
    Catalan               => ("cat_Latn", "ca", "Catalan", ["cat"]),
    Galician              => ("glg_Latn", "gl", "Galician", ["glg"]),
    Occitan               => ("oci_Latn", "oc", "Occitan", ["oci"]),
    Sardinian             => ("srd_Latn", "sc", "Sardinian", ["srd"]),
    Asturian              => ("ast_Latn", "ast", "Asturian", []),
    Hindi                 => ("hin_Deva", "hi", "Hindi", ["hin"]),
    Urdu                  => ("urd_Arab", "ur", "Urdu", ["urd"]),
    Bengali               => ("ben_Beng", "bn", "Bengali", ["ben"]),
    Punjabi               => ("pan_Guru", "pa", "Punjabi", ["pan"]),
    Gujarati              => ("guj_Gujr", "gu", "Gujarati", ["guj"]),
    Marathi               => ("mar_Deva", "mr", "Marathi", ["mar"]),
    Tamil                 => ("tam_Taml", "ta", "Tamil", ["tam"]),
    Telugu                => ("tel_Telu", "te", "Telugu", ["tel"]),
    Kannada               => ("kan_Knda", "kn", "Kannada", ["kan"]),
    Malayalam             => ("mal_Mlym", "ml", "Malayalam", ["mal"]),
    Sinhala               => ("sin_Sinh", "si", "Sinhala", ["sin"]),
    Nepali                => ("npi_Deva", "ne", "Nepali", ["nep", "npi"]),
    Sanskrit              => ("san_Deva", "sa", "Sanskrit", ["san"]),
    Kashmiri              => ("kas_Arab", "ks", "Kashmiri (Arabic script)", ["kas"]),
    KashmiriDevanagari    => ("kas_Deva", "ks-Deva", "Kashmiri (Devanagari)", []),
    Sindhi                => ("snd_Arab", "sd", "Sindhi", ["snd"]),
    Oriya                 => ("ory_Orya", "or", "Oriya", ["ori"]),
    Assamese              => ("asm_Beng", "as", "Assamese", ["asm"]),
    Arabic                => ("arb_Arab", "ar", "Arabic", ["ara", "arb"]),
    Hebrew                => ("heb_Hebr", "he", "Hebrew", ["heb"]),
    Maltese               => ("mlt_Latn", "mt", "Maltese", ["mlt"]),
    Amharic               => ("amh_Ethi", "am", "Amharic", ["amh"]),
    Tigrinya              => ("tir_Ethi", "ti", "Tigrinya", ["tir"]),
    Kabyle                => ("kab_Latn", "kab", "Kabyle", []),
    Tamasheq              => ("taq_Latn", "taq", "Tamasheq (Latin)", []),
    TamasheqTifinagh      => ("taq_Tfng", "taq-Tfng", "Tamasheq (Tifinagh)", []),
    CentralAtlasTamazight => ("tzm_Tfng", "tzm", "Central Atlas Tamazight", []),
    Turkish               => ("tur_Latn", "tr", "Turkish", ["tur"]),
    Azerbaijani           => ("azj_Latn", "az", "Azerbaijani", ["aze", "azj"]),
    Kazakh                => ("kaz_Cyrl", "kk", "Kazakh", ["kaz"]),
    Uzbek                 => ("uzn_Latn", "uz", "Uzbek", ["uzb", "uzn"]),
    Turkmen               => ("tuk_Latn", "tk", "Turkmen", ["tuk"]),
    Kyrgyz                => ("kir_Cyrl", "ky", "Kyrgyz", ["kir"]),
    Tatar                 => ("tat_Cyrl", "tt", "Tatar", ["tat"]),
    CrimeanTatar          => ("crh_Latn", "crh", "Crimean Tatar", []),
    Bashkir               => ("bak_Cyrl", "ba", "Bashkir", ["bak"]),
    Uyghur                => ("uig_Arab", "ug", "Uyghur", ["uig"]),
    Chinese               => ("zho_Hans", "zh", "Chinese (Simplified)", ["zho", "chi", "cmn"]),
    Japanese              => ("jpn_Jpan", "ja", "Japanese", ["jpn"]),
    Korean                => ("kor_Hang", "ko", "Korean", ["kor"]),
    Thai                  => ("tha_Thai", "th", "Thai", ["tha"]),
    Vietnamese            => ("vie_Latn", "vi", "Vietnamese", ["vie"]),
    Indonesian            => ("ind_Latn", "id", "Indonesian", ["ind"]),
    Malay                 => ("zsm_Latn", "ms", "Malay", ["may", "msa", "zsm"]),
    #[serde(alias = "Filipino")]
    Tagalog               => ("tgl_Latn", "tl", "Tagalog (Filipino)", ["tgl", "fil"]),
    Cebuano               => ("ceb_Latn", "ceb", "Cebuano", []),
    Javanese              => ("jav_Latn", "jv", "Javanese", ["jav"]),
    Sundanese             => ("sun_Latn", "su", "Sundanese", ["sun"]),
    Burmese               => ("mya_Mymr", "my", "Burmese", ["mya"]),
    Shan                  => ("shn_Mymr", "shn", "Shan", []),
    Khmer                 => ("khm_Khmr", "km", "Khmer", ["khm"]),
    Lao                   => ("lao_Laoo", "lo", "Lao", ["lao"]),
    Tibetan               => ("bod_Tibt", "bo", "Tibetan", ["bod", "tib"]),
    Dzongkha              => ("dzo_Tibt", "dz", "Dzongkha", ["dzo"]),
    Jingpho               => ("kac_Latn", "kac", "Jingpho", []),
    Swahili               => ("swh_Latn", "sw", "Swahili", ["swa", "swh"]),
    Yoruba                => ("yor_Latn", "yo", "Yoruba", ["yor"]),
    Igbo                  => ("ibo_Latn", "ig", "Igbo", ["ibo"]),
    Hausa                 => ("hau_Latn", "ha", "Hausa", ["hau"]),
    Kanuri                => ("knc_Latn", "kr", "Kanuri (Latin)", ["knc"]),
    KanuriArabic          => ("knc_Arab", "knc-Arab", "Kanuri (Arabic script)", []),
    Zulu                  => ("zul_Latn", "zu", "Zulu", ["zul"]),
    Xhosa                 => ("xho_Latn", "xh", "Xhosa", ["xho"]),
    Swati                 => ("ssw_Latn", "ss", "Swati", ["ssw"]),
    Sotho                 => ("sot_Latn", "st", "Sotho", ["sot"]),
    #[serde(alias = "Setswana")]
    Tswana                => ("tsn_Latn", "tn", "Tswana (Setswana)", ["tsn"]),
    Shona                 => ("sna_Latn", "sn", "Shona", ["sna"]),
    Somali                => ("som_Latn", "so", "Somali", ["som"]),
    Oromo                 => ("gaz_Latn", "om", "Oromo", ["orm", "gaz"]),
    Wolof                 => ("wol_Latn", "wo", "Wolof", ["wol"]),
    Fulani                => ("fuv_Latn", "ff", "Fulani", ["ful", "fuv"]),
    Bambara               => ("bam_Latn", "bm", "Bambara", ["bam"]),
    Lingala               => ("lin_Latn", "ln", "Lingala", ["lin"]),
    Luganda               => ("lug_Latn", "lg", "Luganda", ["lug"]),
    Kinyarwanda           => ("kin_Latn", "rw", "Kinyarwanda", ["kin"]),
    #[serde(alias = "Nyanja")]
    Chichewa              => ("nya_Latn", "ny", "Chichewa (Nyanja)", ["nya"]),
    Tumbuka               => ("tum_Latn", "tum", "Tumbuka", []),
    Malagasy              => ("plt_Latn", "mg", "Malagasy", ["mlg", "plt"]),
    Santali               => ("sat_Beng", "sat", "Santali", []),
    Persian               => ("pes_Arab", "fa", "Persian", ["fas", "per", "pes"]),
    Dari                  => ("prs_Arab", "prs", "Dari", []),
    Pashto                => ("pbt_Arab", "ps", "Pashto", ["pus"]),
    Kurdish               => ("kmr_Latn", "ku", "Kurdish (Northern, Kurmanji)", ["kur", "kmr"]),
    CentralKurdish        => ("ckb_Arab", "ckb", "Central Kurdish (Sorani)", []),
    Georgian              => ("kat_Geor", "ka", "Georgian", ["kat", "geo"]),
    Armenian              => ("hye_Armn", "hy", "Armenian", ["hye", "arm"]),
    Samoan                => ("smo_Latn", "sm", "Samoan", ["smo"]),
    Fijian                => ("fij_Latn", "fj", "Fijian", ["fij"]),
    Maori                 => ("mri_Latn", "mi", "Maori", ["mri", "mao"]),
    Quechua               => ("quy_Latn", "qu", "Quechua", ["que", "quy"]),
    Guarani               => ("grn_Latn", "gn", "Guarani", ["grn"]),
    Aymara                => ("ayr_Latn", "ay", "Aymara", ["aym", "ayr"]),
    Esperanto             => ("epo_Latn", "eo", "Esperanto", ["epo"]),
    Mongolian             => ("khk_Cyrl", "mn", "Mongolian", ["mon", "khk"]),
    Basque                => ("eus_Latn", "eu", "Basque", ["eus", "baq"]),
    Finnish               => ("fin_Latn", "fi", "Finnish", ["fin"]),
    Hungarian             => ("hun_Latn", "hu", "Hungarian", ["hun"]),
    Yiddish               => ("ydd_Hebr", "yi", "Yiddish", ["yid", "ydd"]),
    ArabicEgyptian        => ("arz_Arab", "arz", "Arabic (Egyptian)", []),
    #[serde(alias = "ArabicGulf")]
    ArabicTaizziAdeni     => ("acq_Arab", "acq", "Arabic (Ta'izzi-Adeni)", []),
    ArabicMesopotamian    => ("acm_Arab", "acm", "Arabic (Mesopotamian)", []),
    ArabicNajdi           => ("ars_Arab", "ars", "Arabic (Najdi)", []),
    ArabicLevantine       => ("apc_Arab", "apc", "Arabic (North Levantine)", []),
    ArabicSouthLevantine  => ("ajp_Arab", "ajp", "Arabic (South Levantine)", []),
    ArabicMoroccan        => ("ary_Arab", "ary", "Arabic (Moroccan)", []),
    ArabicTunisian        => ("aeb_Arab", "aeb", "Arabic (Tunisian)", []),
    Bhojpuri              => ("bho_Deva", "bho", "Bhojpuri", []),
    Awadhi                => ("awa_Deva", "awa", "Awadhi", []),
    Chhattisgarhi         => ("hne_Deva", "hne", "Chhattisgarhi", []),
    Magahi                => ("mag_Deva", "mag", "Magahi", []),
    Maithili              => ("mai_Deva", "mai", "Maithili", []),
    Manipuri              => ("mni_Beng", "mni", "Manipuri", []),
    Mizo                  => ("lus_Latn", "lus", "Mizo", []),
    Akan                  => ("aka_Latn", "aka", "Akan", ["ak"]),
    Twi                   => ("twi_Latn", "tw", "Twi", ["twi"]),
    Bemba                 => ("bem_Latn", "bem", "Bemba", []),
    Chokwe                => ("cjk_Latn", "cjk", "Chokwe", []),
    Dinka                 => ("dik_Latn", "dik", "Dinka (Southwestern)", ["din"]),
    Dyula                 => ("dyu_Latn", "dyu", "Dyula", []),
    Nuer                  => ("nus_Latn", "nus", "Nuer", []),
    Ewe                   => ("ewe_Latn", "ewe", "Ewe", ["ee"]),
    Fon                   => ("fon_Latn", "fon", "Fon", []),
    Kabiye                => ("kbp_Latn", "kbp", "Kabiye", []),
    Mossi                 => ("mos_Latn", "mos", "Mossi", []),
    Kikuyu                => ("kik_Latn", "kik", "Kikuyu", ["ki"]),
    Kamba                 => ("kam_Latn", "kam", "Kamba", []),
    Kimbundu              => ("kmb_Latn", "kmb", "Kimbundu", []),
    Kikongo               => ("kon_Latn", "kg", "Kikongo", ["kon"]),
    Umbundu               => ("umb_Latn", "umb", "Umbundu", []),
    #[serde(alias = "Rundi")]
    Kirundi               => ("run_Latn", "run", "Kirundi (Rundi)", ["rn"]),
    Luo                   => ("luo_Latn", "luo", "Luo", []),
    LubaKasai             => ("lua_Latn", "lua", "Luba-Kasai", []),
    #[serde(alias = "Sepedi")]
    NorthernSotho         => ("nso_Latn", "nso", "Northern Sotho (Sepedi)", []),
    Sango                 => ("sag_Latn", "sag", "Sango", ["sg"]),
    Tsonga                => ("tso_Latn", "tso", "Tsonga", ["ts"]),
    Acehnese              => ("ace_Latn", "ace", "Acehnese (Latin)", []),
    AcehneseArabic        => ("ace_Arab", "ace-Arab", "Acehnese (Arabic script)", []),
    Balinese              => ("ban_Latn", "ban", "Balinese", []),
    Banjar                => ("bjn_Latn", "bjn", "Banjar (Latin)", []),
    BanjarArabic          => ("bjn_Arab", "bjn-Arab", "Banjar (Arabic script)", []),
    Buginese              => ("bug_Latn", "bug", "Buginese", []),
    Ilocano               => ("ilo_Latn", "ilo", "Ilocano", []),
    Minangkabau           => ("min_Latn", "min", "Minangkabau", []),
    Pangasinan            => ("pag_Latn", "pag", "Pangasinan", []),
    Waray                 => ("war_Latn", "war", "Waray", []),
    Friulian              => ("fur_Latn", "fur", "Friulian", []),
    Ligurian              => ("lij_Latn", "lij", "Ligurian", []),
    Lombard               => ("lmo_Latn", "lmo", "Lombard", []),
    Sicilian              => ("scn_Latn", "scn", "Sicilian", []),
    Venetian              => ("vec_Latn", "vec", "Venetian", []),
    HaitianCreole         => ("hat_Latn", "hat", "Haitian Creole", ["ht"]),
    Papiamento            => ("pap_Latn", "pap", "Papiamento", []),
    Kabuverdianu          => ("kea_Latn", "kea", "Kabuverdianu", []),
    TokPisin              => ("tpi_Latn", "tpi", "Tok Pisin", []),
    ChineseTraditional    => ("zho_Hant", "zh-Hant", "Chinese (Traditional)", ["zht"]),
    Cantonese             => ("yue_Hant", "yue", "Cantonese", []),
    SouthAzerbaijani      => ("azb_Arab", "azb", "South Azerbaijani (Arabic script)", []),
    Tajik                 => ("tgk_Cyrl", "tg", "Tajik", ["tgk"]),
}

impl NllbLanguage {
    fn info(&self) -> &'static LanguageInfo {
        &LANGUAGES[*self as usize]
    }

    /// Always `true`: every variant has a real NLLB-200 language token.
    ///
    /// Kept for API compatibility. Older versions listed languages NLLB does
    /// not cover (mapped to `eng_Latn`); those entries have been removed.
    pub fn is_nllb_supported(&self) -> bool {
        true
    }

    /// Resolve a user-supplied code.
    ///
    /// Accepts ISO 639-1/639-2/639-3 codes and aliases (`fr`, `fra`, `fil`)
    /// and NLLB/FLORES-200 codes (`fra_Latn`), all case-insensitive. Returns
    /// `None` for anything NLLB-200 does not support.
    pub fn resolve(code: &str) -> Option<Self> {
        let code = code.trim();
        Self::from_iso_code(code).or_else(|| {
            LANGUAGES.iter().find(|l| l.nllb.eq_ignore_ascii_case(code)).map(|l| l.lang)
        })
    }

    /// NLLB/FLORES-200 language code used by the model, e.g. `fra_Latn`.
    pub fn nllb_code(&self) -> &'static str {
        self.info().nllb
    }

    /// Primary short code: ISO 639-1 where one exists, else ISO 639-3 (with a
    /// `-Script` suffix for a language's second script, e.g. `ks-Deva`).
    pub fn iso_code(&self) -> &'static str {
        self.info().iso
    }

    /// Look up by short code or alias (case-insensitive). Does not accept
    /// NLLB codes; use [`NllbLanguage::resolve`] for that.
    pub fn from_iso_code(code: &str) -> Option<Self> {
        LANGUAGES
            .iter()
            .find(|l| l.iso.eq_ignore_ascii_case(code) || l.aliases.iter().any(|a| a.eq_ignore_ascii_case(code)))
            .map(|l| l.lang)
    }

    /// All 202 NLLB-200 languages, in table order.
    pub fn all_languages() -> Vec<Self> {
        LANGUAGES.iter().map(|l| l.lang).collect()
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        self.info().name
    }

    /// SONAR language code. SONAR uses the NLLB/FLORES-200 codes, so this is
    /// identical to [`NllbLanguage::nllb_code`].
    pub fn sonar_code(&self) -> &'static str {
        self.nllb_code()
    }
}

impl fmt::Display for NllbLanguage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// NLLB-200's language tokens, as listed in `FAIRSEQ_LANGUAGE_CODES` of
    /// Hugging Face transformers' `tokenization_nllb.py` (identical to the
    /// pinned `tokenizer.json` and `special_tokens_map.json`).
    const NLLB_LANGUAGE_TOKENS: &[&str] = &[
        "ace_Arab", "ace_Latn", "acm_Arab", "acq_Arab", "aeb_Arab", "afr_Latn", "ajp_Arab", "aka_Latn",
        "als_Latn", "amh_Ethi", "apc_Arab", "arb_Arab", "ars_Arab", "ary_Arab", "arz_Arab", "asm_Beng",
        "ast_Latn", "awa_Deva", "ayr_Latn", "azb_Arab", "azj_Latn", "bak_Cyrl", "bam_Latn", "ban_Latn",
        "bel_Cyrl", "bem_Latn", "ben_Beng", "bho_Deva", "bjn_Arab", "bjn_Latn", "bod_Tibt", "bos_Latn",
        "bug_Latn", "bul_Cyrl", "cat_Latn", "ceb_Latn", "ces_Latn", "cjk_Latn", "ckb_Arab", "crh_Latn",
        "cym_Latn", "dan_Latn", "deu_Latn", "dik_Latn", "dyu_Latn", "dzo_Tibt", "ell_Grek", "eng_Latn",
        "epo_Latn", "est_Latn", "eus_Latn", "ewe_Latn", "fao_Latn", "fij_Latn", "fin_Latn", "fon_Latn",
        "fra_Latn", "fur_Latn", "fuv_Latn", "gaz_Latn", "gla_Latn", "gle_Latn", "glg_Latn", "grn_Latn",
        "guj_Gujr", "hat_Latn", "hau_Latn", "heb_Hebr", "hin_Deva", "hne_Deva", "hrv_Latn", "hun_Latn",
        "hye_Armn", "ibo_Latn", "ilo_Latn", "ind_Latn", "isl_Latn", "ita_Latn", "jav_Latn", "jpn_Jpan",
        "kab_Latn", "kac_Latn", "kam_Latn", "kan_Knda", "kas_Arab", "kas_Deva", "kat_Geor", "kaz_Cyrl",
        "kbp_Latn", "kea_Latn", "khk_Cyrl", "khm_Khmr", "kik_Latn", "kin_Latn", "kir_Cyrl", "kmb_Latn",
        "kmr_Latn", "knc_Arab", "knc_Latn", "kon_Latn", "kor_Hang", "lao_Laoo", "lij_Latn", "lim_Latn",
        "lin_Latn", "lit_Latn", "lmo_Latn", "ltg_Latn", "ltz_Latn", "lua_Latn", "lug_Latn", "luo_Latn",
        "lus_Latn", "lvs_Latn", "mag_Deva", "mai_Deva", "mal_Mlym", "mar_Deva", "min_Latn", "mkd_Cyrl",
        "mlt_Latn", "mni_Beng", "mos_Latn", "mri_Latn", "mya_Mymr", "nld_Latn", "nno_Latn", "nob_Latn",
        "npi_Deva", "nso_Latn", "nus_Latn", "nya_Latn", "oci_Latn", "ory_Orya", "pag_Latn", "pan_Guru",
        "pap_Latn", "pbt_Arab", "pes_Arab", "plt_Latn", "pol_Latn", "por_Latn", "prs_Arab", "quy_Latn",
        "ron_Latn", "run_Latn", "rus_Cyrl", "sag_Latn", "san_Deva", "sat_Beng", "scn_Latn", "shn_Mymr",
        "sin_Sinh", "slk_Latn", "slv_Latn", "smo_Latn", "sna_Latn", "snd_Arab", "som_Latn", "sot_Latn",
        "spa_Latn", "srd_Latn", "srp_Cyrl", "ssw_Latn", "sun_Latn", "swe_Latn", "swh_Latn", "szl_Latn",
        "tam_Taml", "taq_Latn", "taq_Tfng", "tat_Cyrl", "tel_Telu", "tgk_Cyrl", "tgl_Latn", "tha_Thai",
        "tir_Ethi", "tpi_Latn", "tsn_Latn", "tso_Latn", "tuk_Latn", "tum_Latn", "tur_Latn", "twi_Latn",
        "tzm_Tfng", "uig_Arab", "ukr_Cyrl", "umb_Latn", "urd_Arab", "uzn_Latn", "vec_Latn", "vie_Latn",
        "war_Latn", "wol_Latn", "xho_Latn", "ydd_Hebr", "yor_Latn", "yue_Hant", "zho_Hans", "zho_Hant",
        "zsm_Latn", "zul_Latn",
    ];

    #[test]
    fn table_is_exactly_the_nllb_language_set() {
        assert_eq!(NLLB_LANGUAGE_TOKENS.len(), 202);
        let ours: HashSet<_> = LANGUAGES.iter().map(|l| l.nllb).collect();
        let nllb: HashSet<_> = NLLB_LANGUAGE_TOKENS.iter().copied().collect();
        assert_eq!(ours.len(), LANGUAGES.len(), "duplicate NLLB code in table");
        assert_eq!(ours, nllb);
    }

    #[test]
    fn table_index_matches_discriminant() {
        for (i, l) in LANGUAGES.iter().enumerate() {
            assert_eq!(l.lang as usize, i);
        }
    }

    #[test]
    fn codes_and_names_are_unambiguous() {
        let mut keys = HashSet::new();
        for l in LANGUAGES {
            for k in std::iter::once(l.iso).chain(l.aliases.iter().copied()) {
                assert!(keys.insert(k.to_ascii_lowercase()), "code {k} used twice");
            }
        }
        let names: HashSet<_> = LANGUAGES.iter().map(|l| l.name).collect();
        assert_eq!(names.len(), LANGUAGES.len());
    }

    #[test]
    fn every_code_round_trips() {
        for lang in NllbLanguage::all_languages() {
            assert_eq!(NllbLanguage::from_iso_code(lang.iso_code()), Some(lang));
            assert_eq!(NllbLanguage::resolve(lang.nllb_code()), Some(lang));
            assert_eq!(NllbLanguage::resolve(&lang.nllb_code().to_uppercase()), Some(lang));
        }
    }

    #[test]
    fn aliases_and_synonyms_resolve() {
        assert_eq!(NllbLanguage::resolve("fil"), Some(NllbLanguage::Tagalog));
        assert_eq!(NllbLanguage::resolve("nb"), Some(NllbLanguage::Norwegian));
        assert_eq!(NllbLanguage::resolve("nn"), Some(NllbLanguage::NorwegianNynorsk));
        assert_eq!(NllbLanguage::resolve("zh"), Some(NllbLanguage::Chinese));
        assert_eq!(NllbLanguage::resolve(" ks-deva "), Some(NllbLanguage::KashmiriDevanagari));
        let old: NllbLanguage = serde_json::from_str("\"Filipino\"").unwrap();
        assert_eq!(old, NllbLanguage::Tagalog);
        assert_eq!(serde_json::to_string(&old).unwrap(), "\"Tagalog\"");
    }

    #[test]
    fn languages_nllb_lacks_are_absent() {
        // Latin, Bavarian, Corsican, Hawaiian, Low German, West Frisian,
        // Yakut, Chechen: no NLLB-200 token, so not in the table.
        for code in ["la", "bar", "co", "haw", "nds", "fy", "sah", "ce", "sr-latn", "kk-latn"] {
            assert_eq!(NllbLanguage::resolve(code), None, "{code}");
        }
    }

    /// Cross-check against the downloaded tokenizer when it is present.
    #[test]
    fn every_language_has_a_token_in_the_local_tokenizer() {
        let Some(path) = dirs::home_dir().map(|h| h.join(".cache/lingo/nllb-200-distilled-600M/tokenizer.json")) else {
            return;
        };
        let Ok(tokenizer) = tokenizers::Tokenizer::from_file(&path) else {
            return;
        };
        for lang in NllbLanguage::all_languages() {
            assert!(tokenizer.token_to_id(lang.nllb_code()).is_some(), "{}", lang.nllb_code());
        }
    }
}
