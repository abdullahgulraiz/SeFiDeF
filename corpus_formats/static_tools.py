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
    description_elements = parsed_html.find_all(
        "div", class_="markdown-body comment-body p-0"
    )
    scraped_text = ""
    for description_element in description_elements:
        p_texts = [p_element.text for p_element in description_element.find_all("p")]
        scraped_text += " ".join(p_texts)
    return scraped_text


def scrape_data_from_url(url):
    print(f"Scraping data for url: {url}")
    if "https://nvd.nist.gov/" in url:
        result = get_nvd_data(url.split("/")[-1])
    elif "https://github.com/advisories/" in url:
        result = scrape_github_advisory_data(url)
    else:
        return f"Undefined scrape function for url {url}"
    return result


def anchore_get_cve_id_from_nvd_data(nvd_data):
    for obj in nvd_data:
        if "id" in obj:
            return obj["id"]
    return ""


def anchore_get_package_name_from_package_path(package_path: str):
    return package_path.split("/")[-2]


anchore_trivy_description = CorpusFormat(
    name="Anchore/Trivy, url_scraped/description",
    format_dict={
        "keys": [
            {
                "tool": "anchore",
                "fields": ("url",),
                "processing_functions": {"url": scrape_data_from_url},
                "ensure_fields": False,
            },
            {"tool": "trivy", "fields": ("Description",), "ensure_fields": True},
        ],
        "separator": " - ",
    },
)

anchore_trivy_package_name_cve_id_description = CorpusFormat(
    name="Anchore/Trivy, package_name cve_id description",
    format_dict={
        "keys": [
            {
                "tool": "anchore",
                "fields": (
                    "package_path",
                    "nvd_data",
                    "url",
                ),  # ('nvd_data', 'package_path'),
                "processing_functions": {
                    "nvd_data": anchore_get_cve_id_from_nvd_data,
                    "package_path": anchore_get_package_name_from_package_path,
                    "url": scrape_data_from_url,
                },
                "ensure_fields": True,
            },
            {
                "tool": "trivy",
                "fields": ("PkgName", "VulnerabilityID", "Description"),
                "ensure_fields": True,
            },
        ],
        "separator": " ",
    },
)

anchore_trivy_cve_id = CorpusFormat(
    name="Anchore/Trivy, cve_id",
    format_dict={
        "keys": [
            {
                "tool": "anchore",
                "fields": ("nvd_data",),
                "processing_functions": {"nvd_data": anchore_get_cve_id_from_nvd_data},
                "ensure_fields": True,
            },
            {"tool": "trivy", "fields": ("VulnerabilityID",), "ensure_fields": True},
        ],
        "separator": " ",
    },
)

multiple_static_tools_ds_descriptions = CorpusFormat(
    name="Multiple Static Tools, descriptions",
    format_dict={
        "keys": [
            {
                "tool": "anchore",
                # 'fields': ('url',),  # un-comment if scraped data not available for anchore findings (from SeFiLa)
                # 'processing_functions': {'url': scrape_data_from_url},
                "fields": ("scraped_description",),
                "ensure_fields": True,
            },
            {"tool": "trivy", "fields": ("Description",), "ensure_fields": True},
            {
                "tool": "codeql",
                "fields": ("message",),
                "processing_functions": {"message": lambda msg: msg["text"]},
                "ensure_fields": True,
            },
            {
                "tool": "dependency_checker",
                "fields": ("description",),
                "ensure_fields": False,  # TODO: find solution for tool exception categories
            },
            {
                "tool": "gitleaks",
                "fields": ("Description", "Message"),
                "ensure_fields": True,
            },
            {
                "tool": "horusec",
                "fields": ("vulnerabilities",),
                "processing_functions": {
                    "vulnerabilities": lambda vuln: vuln["details"]
                },
                "ensure_fields": True,
            },
            {"tool": "semgrep", "fields": ("message",), "ensure_fields": True},
        ],
        "separator": " ",
    },
)

multiple_static_tools_ds_cve_ids = CorpusFormat(
    name="Multiple Static Tools, cve_ids",
    format_dict={
        "keys": [
            {
                "tool": "anchore",
                "fields": ("nvd_data",),
                "processing_functions": {
                    "nvd_data": lambda nvd_data: nvd_data[0]["id"]
                    if len(nvd_data) > 0
                    else ""
                },
                "ensure_fields": True,
            },
            {"tool": "trivy", "fields": ("VulnerabilityID",), "ensure_fields": True},
            {
                "tool": "dependency_checker",
                "fields": ("name",),
                "ensure_fields": False,  # TODO: find solution for tool exception categories
            },
        ],
        "separator": " ",
    },
)
