<?xml version="1.0" encoding="ISO-8859-1"?>
<xsl:stylesheet version="1.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:pm="http://www.tobedefined.com/pampelmuse">

<!--<xsl:output method="html"/>-->

<xsl:template match="/">
<html>
  <body>

    <table width="1000" border="0">
      <tr>
	<td colspan="2" style="background-color:#FFA500;" align="center">
	  <h1>PampelMuse configuration sheet</h1>
	  <i>
	    last updated: <xsl:value-of select="/pm:configuration/@timestamp"/>
	  </i>
	</td>
      </tr>
      <xsl:for-each select="/pm:configuration/pm:parameter[@inuse='1']">
	<xsl:sort select="@category"/>
	<tr>
	  <td style="background-color:#FFD700;width:100px;text-align:top;">
	    <b>
	    <xsl:value-of select="pm:name"/>
	    </b>
	  </td>
	  <td style="background-color:#EEEEEE;height:50px;width:400px;
		     text-align:top;color:#0000FF" align="left">
	    <i><xsl:value-of select="pm:value"/></i>
<!--	    <xsl:value-of select="pm:float"/>
	    <xsl:value-of select="pm:string"/>
	    <xsl:value-of select="pm:integer"/>
	    <xsl:value-of select="pm:flag"/>      -->
	  </td>
	</tr>
	<tr>
	  <td style="background-color:#EEEEEE;" align="center"  valign="top">
	    <i><xsl:value-of select="./pm:value/@dtype"/></i>
	  </td>
	  <td style="background-color:#EEEEEE;">
	    <xsl:for-each select="./pm:comment">
	      <xsl:value-of select="."/><br />
	    </xsl:for-each>
	  </td>
	</tr>
	</xsl:for-each>
	<tr><td colspan="3" style="height=5px"/></tr>
    </table>

  </body>
</html>
</xsl:template>




<!--<xsl:template match="@pm:dtype">
  <xsl:attribute name="pm:dtype">
    <td style="background-color:#EEEEEE;height:50px;width:200px;
	       text-align:top;" align="center">
      <xsl:value-of select="@pm:dtype"/>
    </td>
  </xsl:attribute>
</xsl:template>
-->

</xsl:stylesheet>
