from corpus_formats.base import CorpusFormat

name_description_solution_1 = CorpusFormat(
    name="ZAP/Arachni, name description solution",
    format_dict={
        'keys': [{'tool': 'arachni',
                  'fields': ('name', 'description', 'remedy_guidance'),
                  'ensure_fields': False},
                 {'tool': 'zap',
                  'fields': ('name', 'desc', 'solution'),
                  'ensure_fields': False}],
        'separator': " "
    }
)