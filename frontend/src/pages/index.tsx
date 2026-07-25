import * as React from "react";
import HALOAppLayout from "../components/layout";
import { graphql } from "gatsby";

// markup
const IndexPage = ({ data }: any) => {
  return (
    <HALOAppLayout meta={data.site.siteMetadata} title="Home" link={"/"}>
      <main style={{ height: "100%" }} className=" h-full ">
      </main>
    </HALOAppLayout>
  );
};

export const query = graphql`
  query HomePageQuery {
    site {
      siteMetadata {
        description
        title
      }
    }
  }
`;

export default IndexPage;
