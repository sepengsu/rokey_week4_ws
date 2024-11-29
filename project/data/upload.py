import requests
import spb_curate

spb_curate.team_name = "some-team" # Suite(Superb Curate) team name
spb_curate.access_key = "some-team-access-key" # team access key

new_dataset = spb_curate.create_dataset(
    name="Augmented dataset",
    description="Test dataset for demo", # Add descriptions to the dataset
)
path = "image.jpg"
job = new_dataset.add_images(
    images=[
        spb_curate.Image(
            key="77396",
            source=spb_curate.ImageSourceLocal(
                asset= path,
                asset_id="000000077396",
            ),
        )
    ]
)

spb_curate.ImageSourceLocal(

)