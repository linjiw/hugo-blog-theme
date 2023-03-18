---
title: Colorizing the Prokudin-Gorskii Photo Collection
subtitle: The Vibrant Transformation of Prokudin-Gorskii's Photo Archive

# Summary for listings and search engines
summary: This project requires you to use image processing techniques to create a color image from the digitized Prokudin-Gorskii glass plate photographs, with the goal of producing an image with as few visual artifacts as possible. A digital picture of a glass plate, with its three channels arranged from top to bottom as BGR (see Figure 1), serves as the process's input. Our task is to isolate the three channels and properly align them.

# Link this post with a project
projects: []

# Date published
date: "2020-12-13T00:00:00Z"

# Date updated
lastmod: "2020-12-13T00:00:00Z"

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: false

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: 'Image credit: Prokudin-Gorskii Collection'
  focal_point: ""
  placement: 2
  preview_only: false

authors:
- admin
# - 美杨

tags:
- Computer Vision
- Image Editing
# - 开源

categories:
- Project
# - 教程
---

## Matching Metric

Image matching means to manipulate image to ensure the best results for your similarity metrics. Thus, it is important to choose an efficient and accurate image matching metrics. For this assignment, I explored Sum of Squared Differences (SSD) and Normalized Cross Correlation (NCC) functions. The objective is to maximize or minimize one of these functions by searching and manipulating the images.

**Sum of Squared Differences** SSD  is calculated     based on the following equation:

{{< figure src="./data/ssd.png" title="SSD Equation" >}}

where *I* and  *H* are function of two images and *x*,*y* are pixels positions.

{{< figure src="./data/11_13PM_February_14_2023/cathedral.jpg_square_limit25_CROPFalse_CannyTrue_PyramidFalse_CMEASFalse_methodSSD_shift[[0, -25],[7, -25]]_time_cost 2.1364333629608154.jpg" title="Figure 2. Cathedral: SSD, Shift:[0, -25],[7, -25] time cost: 2.13s" >}}



**Normalized Cross Correlation** NNC is calculated     based on the following equation: 

{{< figure src="./data/ncc.png" title="NCC Equation" >}}


## Get Started

- 👉 [**Create a new site**](https://wowchemy.com/hugo-themes/)
- 📚 [**Personalize your site**](https://wowchemy.com/docs/)
- 💬 [Chat with the **Wowchemy community**](https://discord.gg/z8wNYzb) or [**Hugo community**](https://discourse.gohugo.io)
- 🐦 Twitter: [@wowchemy](https://twitter.com/wowchemy) [@GeorgeCushen](https://twitter.com/GeorgeCushen) [#MadeWithWowchemy](https://twitter.com/search?q=(%23MadeWithWowchemy%20OR%20%23MadeWithAcademic)&src=typed_query)
- 💡 [Request a **feature** or report a **bug** for _Wowchemy_](https://github.com/wowchemy/wowchemy-hugo-themes/issues)
- ⬆️ **Updating Wowchemy?** View the [Update Guide](https://wowchemy.com/docs/hugo-tutorials/update/) and [Release Notes](https://github.com/wowchemy/wowchemy-hugo-themes/releases)

## Crowd-funded open-source software

To help us develop this template and software sustainably under the MIT license, we ask all individuals and businesses that use it to help support its ongoing maintenance and development via sponsorship.

### [❤️ Click here to become a sponsor and help support Wowchemy's future ❤️](https://github.com/sponsors/gcushen)

As a token of appreciation for sponsoring, you can **unlock [these](https://wowchemy.com/sponsor/) awesome rewards and extra features 🦄✨**

## Ecosystem

* **[Hugo Academic CLI](https://github.com/wowchemy/hugo-academic-cli):** Automatically import publications from BibTeX

## Inspiration

[Check out the latest **demo**](https://hugo-blog-theme.netlify.app/) of what you'll get in less than 10 minutes, or [view the **showcase**](https://wowchemy.com/creators/) of personal, project, and business sites.

## Features

- **Page builder** - Create *anything* with [**widgets**](https://wowchemy.com/docs/page-builder/) and [**elements**](https://wowchemy.com/docs/writing-markdown-latex/)
- **Edit any type of content** - Blog posts, publications, talks, slides, projects, and more!
- **Create content** in [**Markdown**](https://wowchemy.com/docs/writing-markdown-latex/), [**Jupyter**](https://wowchemy.com/docs/import/jupyter/), or [**RStudio**](https://wowchemy.com/docs/install-locally/)
- **Plugin System** - Fully customizable [**color** and **font themes**](https://wowchemy.com/docs/customization/)
- **Display Code and Math** - Code highlighting and [LaTeX math](https://en.wikibooks.org/wiki/LaTeX/Mathematics) supported
- **Integrations** - [Google Analytics](https://analytics.google.com), [Disqus commenting](https://disqus.com), Maps, Contact Forms, and more!
- **Beautiful Site** - Simple and refreshing one page design
- **Industry-Leading SEO** - Help get your website found on search engines and social media
- **Media Galleries** - Display your images and videos with captions in a customizable gallery
- **Mobile Friendly** - Look amazing on every screen with a mobile friendly version of your site
- **Multi-language** - 34+ language packs including English, 中文, and Português
- **Multi-user** - Each author gets their own profile page
- **Privacy Pack** - Assists with GDPR
- **Stand Out** - Bring your site to life with animation, parallax backgrounds, and scroll effects
- **One-Click Deployment** - No servers. No databases. Only files.

## Themes

Wowchemy and its templates come with **automatic day (light) and night (dark) mode** built-in. Alternatively, visitors can choose their preferred mode - click the moon icon in the top right of the [Demo](https://academic-demo.netlify.com/) to see it in action! Day/night mode can also be disabled by the site admin in `params.toml`.

[Choose a stunning **theme** and **font**](https://wowchemy.com/docs/customization) for your site. Themes are fully customizable.

## License

Copyright 2016-present [George Cushen](https://georgecushen.com).

Released under the [MIT](https://github.com/wowchemy/wowchemy-hugo-themes/blob/master/LICENSE.md) license.
