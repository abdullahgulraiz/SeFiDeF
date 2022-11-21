from corpus_formats.base import CorpusFormat

zap_arachni_name_description_solution = CorpusFormat(
    name="ZAP/Arachni, name description solution",
    format_dict={
        "keys": [
            {
                "tool": "arachni",
                "fields": ("name", "description", "remedy_guidance"),
                "ensure_fields": False,
            },
            {
                "tool": "zap",
                "fields": ("name", "desc", "solution"),
                "ensure_fields": False,
            },
        ],
        "separator": " ",
    },
)

zap_arachni_description_solution = CorpusFormat(
    name="ZAP/Arachni, description solution",
    format_dict={
        "keys": [
            {
                "tool": "arachni",
                "fields": ("description", "remedy_guidance"),
                "ensure_fields": False,
            },
            {"tool": "zap", "fields": ("desc", "solution"), "ensure_fields": False},
        ],
        "separator": " ",
    },
)

zap_arachni_description = CorpusFormat(
    name="ZAP/Arachni, description",
    format_dict={
        "keys": [
            {"tool": "arachni", "fields": ("description",), "ensure_fields": True},
            {"tool": "zap", "fields": ("desc",), "ensure_fields": True},
        ],
        "separator": " ",
    },
)
