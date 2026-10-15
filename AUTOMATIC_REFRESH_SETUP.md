# Automatic results refresh

This package adds a daily GitHub Actions workflow and a scraper for the public LeagueTables U14 Girls GLA East/West pages.

## Install once
1. Upload the files in this package to the root of `RCashdollar/ecnl-gla-u14g-predictor`, replacing the existing files when prompted.
2. Make sure the `.github/workflows/refresh-results.yml` file is included. GitHub may require enabling Actions for the repository.
3. Open the repository's **Actions** tab, select **Refresh GLA U14 Girls results**, and choose **Run workflow** to test it.
4. Open the run and check the log. The scraper intentionally stops without changing `results.csv` if either source fails or fewer than five scored matches can be recognized. If that happens, the page's HTML structure needs a parser adjustment before relying on scheduled imports.
5. If the run succeeds, the workflow commits `results.csv` only when it changes. Streamlit Community Cloud will redeploy after that commit.

The workflow runs daily at 14:17 UTC and can also be run manually. It does not require a personal access token; GitHub Actions uses the repository's built-in token.

## Important limitations
- LeagueTables says its tables are computed from displayed results and recommends checking the league or club for official corrections.
- This is a conservative first-pass parser. Review the first run carefully before treating imported results as complete.
- Existing rows are retained. User-confirmed rows are protected from score replacement by the scraper.
