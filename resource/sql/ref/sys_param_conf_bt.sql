CREATE DATABASE  IF NOT EXISTS `mes_omi` /*!40100 DEFAULT CHARACTER SET utf8 */;
USE `mes_omi`;
-- MySQL dump 10.13  Distrib 5.7.12, for Win32 (AMD64)
--
-- Host: 127.0.0.1    Database: mes_omi
-- ------------------------------------------------------
-- Server version	5.5.5-10.2.7-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `sys_param_conf_bt`
--

DROP TABLE IF EXISTS `sys_param_conf_bt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `sys_param_conf_bt` (
  `param_id` varchar(8) NOT NULL,
  `desc` varchar(64) DEFAULT NULL COMMENT 'description',
  `cate1` varchar(24) DEFAULT NULL,
  `cate2` varchar(24) DEFAULT NULL,
  `cate3` varchar(24) DEFAULT NULL,
  `value1` varchar(64) DEFAULT NULL,
  `value2` varchar(64) DEFAULT NULL,
  `value3` varchar(64) DEFAULT NULL,
  `value9` varchar(255) DEFAULT NULL,
  `ivalue1` int(11) DEFAULT NULL,
  `ivalue2` int(11) DEFAULT NULL,
  `ivalue3` int(11) DEFAULT NULL,
  `ivalue4` int(11) DEFAULT NULL,
  `fvalue1` float DEFAULT NULL,
  `fvalue2` float DEFAULT NULL,
  `fvalue3` float DEFAULT NULL,
  `note1` varchar(255) DEFAULT NULL,
  `rec_user` varchar(16) DEFAULT NULL,
  `rec_time` varchar(19) DEFAULT NULL,
  PRIMARY KEY (`param_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 AVG_ROW_LENGTH=5461 ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `sys_param_conf_bt`
--

LOCK TABLES `sys_param_conf_bt` WRITE;
/*!40000 ALTER TABLE `sys_param_conf_bt` DISABLE KEYS */;
INSERT INTO `sys_param_conf_bt` VALUES ('2DL-01','auto create laser 2d-code',NULL,NULL,NULL,'','',NULL,NULL,5,1,NULL,NULL,NULL,NULL,NULL,' ivalue1:length ivalue1:cum number ',NULL,NULL),('BOX-01','auto create box_no',NULL,NULL,NULL,NULL,NULL,NULL,NULL,5,1,NULL,NULL,NULL,NULL,NULL,' ivalue1:length ivalue1:cum number ',NULL,NULL),('BOX-90','box_no dummy setting',NULL,NULL,NULL,'Z',NULL,NULL,NULL,5,1461,NULL,NULL,NULL,NULL,NULL,'dummy box,  ivalue1:length ivalue1:cum number ',NULL,NULL),('FLOW-01','wafer start / end step',NULL,NULL,NULL,'065.020','888.880',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL),('LOT-01','auto create lot_id',NULL,NULL,NULL,'A','01',NULL,NULL,6,381,NULL,NULL,NULL,NULL,NULL,'value2:node_id ivalue1:length ivalue1:cum number ',NULL,NULL),('WAF-01','auto create wafer_id',NULL,NULL,NULL,'','',NULL,NULL,5,1,NULL,NULL,NULL,NULL,NULL,' ivalue1:length ivalue1:cum number ',NULL,NULL);
/*!40000 ALTER TABLE `sys_param_conf_bt` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:30
