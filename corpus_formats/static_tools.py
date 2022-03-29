import requests
from bs4 import BeautifulSoup
from corpus_formats.base import CorpusFormat


def get_nvd_data(cve_id):
    query_url = f"https://services.nvd.nist.gov/rest/json/cve/1.0/{cve_id}"
    response_json = requests.get(query_url).json()
    result = []
    # get descriptions for each cve item in result
    for cve_item in response_json["result"]["CVE_Items"]:
        description_temp = []
        # multiple description objects exist, so aggregate information from them
        for description_data in cve_item["cve"]["description"]["description_data"]:
            description_temp.append(description_data["value"])
        result.append(". ".join(description_temp))
    # return aggregate of all descriptions as a single sentence
    return ". ".join(result)


def scrape_github_advisory_data(url):
    response = requests.get(url)
    parsed_html = BeautifulSoup(response.content, "html.parser")
    description_elements = parsed_html.find_all("div", class_="markdown-body comment-body p-0")
    # get first paragraph element
    return description_elements[0].find("p").text


def scrape_data_from_url(url):
    print(f"Scraping data for url: {url}")
    if "https://nvd.nist.gov/" in url:
        result = get_nvd_data(url.split("/")[-1])
    elif "https://github.com/advisories/" in url:
        result = scrape_github_advisory_data(url)
    else:
        return f"Undefined scrape function for url {url}"
    return result


anchore_trivy_name_title_description = CorpusFormat(
    name="Anchore/Trivy, description",
    format_dict={
        'keys': [{'tool': 'anchore',
                  'fields': ('url',),
                  'processing_functions': {'url': scrape_data_from_url},
                  'ensure_fields': False},
                 {'tool': 'trivy',
                  'fields': ('Description', ),
                  'ensure_fields': True}],
        'separator': " - "
    }
)
