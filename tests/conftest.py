import os
import base64
import pytest
from langchain_core.runnables.config import RunnableConfig
from langchain.schema import Document
from unittest.mock import Mock
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from src.db import SqlAlchemyBase
from src.config import Config
from src.api.app import app
from src.api.deps import get_config


load_dotenv()


@pytest.fixture(scope="session")
def average_insurance_expenditures_html() -> str:
    return """
<html>
  <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
  </head>
  <body>
    <table>
<thead><tr><th></th></tr></thead>    <tbody> <tr class="odd"><td><table>
<tr><td>Average Expenditures for Auto Insurance, 2012-2021</td></tr>
<tr><td><p> </p>
</td></tr>
<tr><td><table>
<thead>
<tr>
<th>Year</th>
<th>Average expenditure</th>
<th>Percent change</th>
</tr>
</thead>
<tbody>
<tr>
<td>2012</td>
<td>$812.40</td>
<td>2.2%</td>
</tr>
<tr>
<td>2013</td>
<td>841.06</td>
<td>3.5</td>
</tr>
<tr>
<td>2014</td>
<td>869.47</td>
<td>3.4</td>
</tr>
<tr>
<td>2015</td>
<td>896.66</td>
<td>3.1</td>
</tr>
<tr>
<td>2016</td>
<td>945.22</td>
<td>5.4</td>
</tr>
<tr>
<td>2017</td>
<td>1,008.35</td>
<td>6.7</td>
</tr>
<tr>
<td>2018</td>
<td>1,058.10</td>
<td>4.9</td>
</tr>
<tr>
<td>2019</td>
<td>1,071.74</td>
<td>1.3</td>
</tr>
<tr>
<td>2020</td>
<td>1,046.37</td>
<td>-2.4</td>
</tr>
<tr>
<td>2021</td>
<td>1,061.54</td>
<td>1.4</td>
</tr>
</tbody></table>

Source: National Association of Insurance Commissioners (NAIC). Further reprint or distribution strictly prohibited without written permission of NAIC.</td></tr>
</table></td> </tr>
      </tbody>
    </table>
  </body>
</html>
"""


@pytest.fixture
def average_insurance_expenditures_html_as_data_url(
    average_insurance_expenditures_html,
) -> str:
    """Returns the sample HTML content as a base64 encoded data URI"""
    encoded_html = base64.b64encode(
        average_insurance_expenditures_html.encode("utf-8")
    ).decode("utf-8")
    return f"data:text/html;base64,{encoded_html}"


@pytest.fixture(scope="session")
def premiums_by_state_html() -> str:
    return """
<html>
  <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
  </head>
  <body>
    <table>
    <thead><tr><th></th></tr></thead>    <tbody> <tr class="odd"><td><table>
<tr><td>Direct Premiums Written, P/C Insurance By State, 2023 (1)</td></tr>
<tr><td><p>($000)</p>
</td></tr>
<tr><td><table>
<thead>
<tr>
<th>State</th>
<th>Total, all lines</th>
<th>State</th>
<th>Total, all lines</th>
</tr>
</thead>
<tbody>
<tr>
<td>Alabama</td>
<td>$13,146,853</td>
<td>Montana</td>
<td>$3,807,820</td>
</tr>
<tr>
<td>Alaska</td>
<td>1,994,750</td>
<td>Nebraska</td>
<td>7,573,101</td>
</tr>
<tr>
<td>Arizona</td>
<td>17,959,495</td>
<td>Nevada</td>
<td>8,561,822</td>
</tr>
<tr>
<td>Arkansas</td>
<td>7,931,849</td>
<td>New Hampshire</td>
<td>3,342,091</td>
</tr>
<tr>
<td>California</td>
<td>111,243,502</td>
<td>New Jersey</td>
<td>29,106,640</td>
</tr>
<tr>
<td>Colorado</td>
<td>20,582,327</td>
<td>New Mexico</td>
<td>5,038,116</td>
</tr>
<tr>
<td>Connecticut</td>
<td>11,508,412</td>
<td>New York</td>
<td>61,861,773</td>
</tr>
<tr>
<td>D.C.</td>
<td>2,699,081</td>
<td>North Carolina</td>
<td>24,036,512</td>
</tr>
<tr>
<td>Delaware</td>
<td>3,842,583</td>
<td>North Dakota</td>
<td>3,826,625</td>
</tr>
<tr>
<td>Florida</td>
<td>88,067,510</td>
<td>Ohio</td>
<td>23,397,453</td>
</tr>
<tr>
<td>Georgia</td>
<td>32,039,933</td>
<td>Oklahoma</td>
<td>11,325,023</td>
</tr>
<tr>
<td>Hawaii</td>
<td>3,490,752</td>
<td>Oregon</td>
<td>10,510,197</td>
</tr>
<tr>
<td>Idaho</td>
<td>4,757,044</td>
<td>Pennsylvania</td>
<td>32,532,786</td>
</tr>
<tr>
<td>Illinois</td>
<td>35,938,656</td>
<td>Rhode Island</td>
<td>3,361,749</td>
</tr>
<tr>
<td>Indiana</td>
<td>16,321,562</td>
<td>South Carolina</td>
<td>15,078,257</td>
</tr>
<tr>
<td>Iowa</td>
<td>9,843,752</td>
<td>South Dakota</td>
<td>3,970,217</td>
</tr>
<tr>
<td>Kansas</td>
<td>9,565,831</td>
<td>Tennessee</td>
<td>17,442,243</td>
</tr>
<tr>
<td>Kentucky</td>
<td>10,147,626</td>
<td>Texas</td>
<td>92,331,270</td>
</tr>
<tr>
<td>Louisiana</td>
<td>16,226,096</td>
<td>Utah</td>
<td>8,100,467</td>
</tr>
<tr>
<td>Maine</td>
<td>3,316,037</td>
<td>Vermont</td>
<td>1,640,326</td>
</tr>
<tr>
<td>Maryland</td>
<td>16,186,978</td>
<td>Virginia</td>
<td>20,316,519</td>
</tr>
<tr>
<td>Massachusetts</td>
<td>21,453,245</td>
<td>Washington</td>
<td>18,109,893</td>
</tr>
<tr>
<td>Michigan</td>
<td>24,427,664</td>
<td>West Virginia</td>
<td>3,643,445</td>
</tr>
<tr>
<td>Minnesota</td>
<td>17,081,121</td>
<td>Wisconsin</td>
<td>14,623,479</td>
</tr>
<tr>
<td>Mississippi</td>
<td>7,505,441</td>
<td>Wyoming</td>
<td>1,820,520</td>
</tr>
<tr>
<td>Missouri</td>
<td>17,261,874</td>
<td><strong>United States</strong></td>
<td><strong>$949,898,320</strong></td>
</tr>
</tbody></table>

(1) Before reinsurance transactions, includes state funds, excludes territories.
(2) Data for the total United States may differ from similar data shown elsewhere due to the use of different exhibits from S&amp;P Global Market Intelligence.

Source: NAIC data, sourced from S&amp;P Global Market Intelligence, Insurance Information Institute.</td></tr>
</table></td> </tr>
      </tbody>
    </table>
  </body>
</html>
"""


@pytest.fixture
def premiums_by_state_html_as_data_url(premiums_by_state_html) -> str:
    """Returns the sample HTML content as a base64 encoded data URI"""
    encoded_html = base64.b64encode(premiums_by_state_html.encode("utf-8")).decode(
        "utf-8"
    )
    return f"data:text/html;base64,{encoded_html}"


@pytest.fixture
def mock_db():
    db = Mock()
    context = Mock()
    db.begin.return_value = context
    context.__enter__ = Mock()
    context.__exit__ = Mock()
    return db


@pytest.fixture
def sqlite_session():
    """Create an in-memory SQLite database session"""
    engine = create_engine("sqlite:///:memory:")
    SqlAlchemyBase.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


@pytest.fixture
def tmp_db_url(tmp_path):
    """Create a temporary SQLite database file that gets cleaned up after use"""
    db_url = f"sqlite:///{tmp_path}/test.db"
    engine = create_engine(db_url)
    SqlAlchemyBase.metadata.create_all(engine)

    yield db_url

    engine.dispose()


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization"],
        # openweathermap
        "filter_query_parameters": ["APPID"],
        "ignore_hosts": ["testserver"],
    }


@pytest.fixture
def insurance_data_documents():
    return [
        Document(page_content=content, id=str(i))
        for i, content in enumerate(
            [
                "Average Expenditures for Auto Insurance, 2012-2014\n\n"
                + "Year: 2012, Expenditure: $812.40, Change: 2.2%\n"
                + "Year: 2013, Expenditure: $841.06, Change: 3.5%\n"
                "Year: 2014, Expenditure: $869.47, Change: 3.4%\n",
                "Average Expenditures for Auto Insurance, 2015-2017\n\n"
                + "Year: 2015, Expenditure: $896.66, Change: 3.1%\n"
                + "Year: 2016, Expenditure: $945.22, Change: 5.4%\n"
                + "Year: 2017, Expenditure: $1008.35, Change: 6.7%\n",
                "Average Expenditures for Auto Insurance, 2018-2021\n\n"
                + "Year: 2018, Expenditure: $1058.10, Change: 4.9%\n"
                + "Year: 2019, Expenditure: $1071.74, Change: 1.3%\n"
                + "Year: 2020, Expenditure: $1046.37, Change: -2.4%\n"
                + "Year: 2021, Expenditure: $1061.54, Change: 1.4%\n",
            ]
        )
    ]


@pytest.fixture
def travel_knowledge_documents():
    return [
        Document(page_content=content, id=f"knowledge_base[{i}]")
        for i, content in enumerate(
            [
                "Paris, the 'City of Light,' boasts iconic landmarks such as the Eiffel Tower, offering panoramic views from its observation decks. The Louvre Museum, home to masterpieces like the Mona Lisa, attracts art enthusiasts worldwide. Notre-Dame Cathedral, a Gothic architectural marvel, stands on the Île de la Cité. The Champs-Élysées, lined with shops and cafes, leads to the Arc de Triomphe, honoring those who fought for France. Montmartre, with its artistic heritage, features the Basilica of the Sacré-Cœur atop its hill.",
                "London, a vibrant metropolis, features the historic Tower of London, housing the Crown Jewels. The British Museum showcases a vast collection of world art and artifacts. Buckingham Palace, the monarch's residence, is famed for the Changing of the Guard ceremony. The Houses of Parliament and Big Ben are iconic symbols along the River Thames. The London Eye offers a rotating perspective of the city's skyline.",
                "Rome, the 'Eternal City,' is home to the Colosseum, an ancient amphitheater echoing gladiatorial history. The Vatican City enclaves St. Peter's Basilica and the Sistine Chapel, adorned with Michelangelo's frescoes. The Pantheon, with its impressive dome, stands as a testament to Roman engineering. The Roman Forum's ruins narrate tales of imperial grandeur. Trevi Fountain invites visitors to toss a coin, ensuring their return to Rome.",
                "Berlin, Germany's capital, presents the Brandenburg Gate, a neoclassical monument symbolizing unity. The Berlin Wall Memorial commemorates the city's divided past. Museum Island hosts a cluster of renowned museums, including the Pergamon Museum. The Reichstag Building, with its modern glass dome, offers insights into German politics. Checkpoint Charlie stands as a relic of Cold War history.",
                "Madrid, Spain's lively capital, features the Prado Museum, exhibiting European art masterpieces. The Royal Palace impresses with its opulent architecture and historic significance. Retiro Park provides a green oasis in the city's heart. Puerta del Sol serves as a bustling public square and focal point for festivities. The Gran Vía is renowned for shopping, theaters, and vibrant nightlife.",
            ]
        )
    ]


@pytest.fixture
def config(tmp_path, tmp_db_url) -> Config:
    return Config(
        **{
            "llm": {
                "type": "openai",
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "weather": {
                "api_key": os.getenv("OPENWEATHERMAP_API_KEY"),
            },
            "db": {
                "url": tmp_db_url,
            },
            "embeddings": {
                "type": "chroma-internal",
            },
            "vector_store": {
                "type": "chroma",
                "collection_name": "documents",
                "path": f"{tmp_path}/chroma",
            },
            # ignoring type errors here due to https://docs.pydantic.dev/latest/integrations/visual_studio_code/#strict-errors
        }  # type: ignore
    )


@pytest.fixture
def api_client(config):
    def mock_override_get_config():
        return config

    app.dependency_overrides[get_config] = mock_override_get_config

    client = TestClient(app)

    yield client

    app.dependency_overrides[get_config] = get_config
