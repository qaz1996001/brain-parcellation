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
-- Table structure for table `dcop_param_tool_bt`
--

DROP TABLE IF EXISTS `dcop_param_tool_bt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `dcop_param_tool_bt` (
  `tool_id` varchar(12) NOT NULL COMMENT 'length extend to 12 for sintering "before/after" omi -- 2020/03/18 lkchena',
  `ws_type` varchar(12) NOT NULL DEFAULT '0' COMMENT 'change to use string, compatible with aruroal 2020/03/29 lkchena\n2020/03/17 lkchena\n',
  `cate1` varchar(24) NOT NULL DEFAULT '*',
  `cate2` varchar(24) NOT NULL DEFAULT '*',
  `cate3` varchar(24) NOT NULL DEFAULT '*',
  `active1a` int(11) DEFAULT 0,
  `value1a` varchar(64) DEFAULT NULL,
  `value1b` varchar(64) DEFAULT NULL,
  `value1c` varchar(64) DEFAULT NULL,
  `rate1a` float DEFAULT NULL,
  `active2a` int(11) DEFAULT 0,
  `value2a` varchar(64) DEFAULT NULL,
  `value2b` varchar(64) DEFAULT NULL,
  `value2c` varchar(64) DEFAULT NULL,
  `rate2a` float DEFAULT 1,
  `active3a` int(11) DEFAULT 0,
  `value3a` varchar(64) DEFAULT NULL,
  `value3b` varchar(64) DEFAULT NULL,
  `value3c` varchar(64) DEFAULT NULL,
  `rate3a` float DEFAULT 1,
  `active4a` int(11) DEFAULT 0,
  `value4a` varchar(64) DEFAULT NULL,
  `value4b` varchar(64) DEFAULT NULL,
  `value4c` varchar(64) DEFAULT NULL,
  `rate4a` float DEFAULT 1,
  `active5a` int(11) DEFAULT 0,
  `value5a` varchar(64) DEFAULT NULL,
  `value5b` varchar(64) DEFAULT NULL,
  `value5c` varchar(64) DEFAULT NULL,
  `rate5a` float DEFAULT 1,
  `active6a` int(11) DEFAULT 0,
  `value6a` varchar(64) DEFAULT NULL,
  `value6b` varchar(64) DEFAULT NULL,
  `value6c` varchar(64) DEFAULT NULL,
  `rate6a` float DEFAULT 1,
  `active7a` int(11) DEFAULT 0,
  `value7a` varchar(64) DEFAULT NULL,
  `value7b` varchar(64) DEFAULT NULL,
  `value7c` varchar(64) DEFAULT NULL,
  `rate7a` float DEFAULT 1,
  `active8a` int(11) DEFAULT 0,
  `value8a` varchar(64) DEFAULT NULL,
  `value8b` varchar(64) DEFAULT NULL,
  `value8c` varchar(64) DEFAULT NULL,
  `rate8a` float DEFAULT NULL,
  `active9a` int(11) DEFAULT 0,
  `value9a` varchar(64) DEFAULT NULL,
  `value9b` varchar(64) DEFAULT NULL,
  `value9c` varchar(64) DEFAULT NULL,
  `rate9a` float DEFAULT 1,
  `active10a` int(11) DEFAULT 0,
  `value10a` varchar(64) DEFAULT NULL,
  `value10b` varchar(64) DEFAULT NULL,
  `value10c` varchar(64) DEFAULT NULL,
  `rate10a` float DEFAULT NULL,
  `active11a` int(11) DEFAULT 0,
  `value11a` varchar(64) DEFAULT NULL,
  `value11b` varchar(64) DEFAULT NULL,
  `value11c` varchar(64) DEFAULT NULL,
  `rate11a` float DEFAULT 1,
  `active12a` int(11) DEFAULT 0,
  `value12a` varchar(64) DEFAULT NULL,
  `value12b` varchar(64) DEFAULT NULL,
  `value12c` varchar(64) DEFAULT NULL,
  `rate12a` float DEFAULT NULL,
  `active13a` int(11) DEFAULT 0,
  `value13a` varchar(64) DEFAULT NULL,
  `value13b` varchar(64) DEFAULT NULL,
  `value13c` varchar(64) DEFAULT NULL,
  `rate13a` float DEFAULT 1,
  `active14a` int(11) DEFAULT 0,
  `value14a` varchar(64) DEFAULT NULL,
  `value14b` varchar(64) DEFAULT NULL,
  `value14c` varchar(64) DEFAULT NULL,
  `rate14a` float DEFAULT NULL,
  `active15a` int(11) DEFAULT 0,
  `value15a` varchar(64) DEFAULT NULL,
  `value15b` varchar(64) DEFAULT NULL,
  `value15c` varchar(64) DEFAULT NULL,
  `rate15a` float DEFAULT 1,
  `active16a` int(11) DEFAULT 0,
  `value16a` varchar(64) DEFAULT NULL,
  `value16b` varchar(64) DEFAULT NULL,
  `value16c` varchar(64) DEFAULT NULL,
  `rate16a` float DEFAULT 1,
  `note1` varchar(255) DEFAULT NULL,
  `rec_user` varchar(16) DEFAULT NULL,
  `rec_time` varchar(19) DEFAULT NULL,
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6),
  PRIMARY KEY (`tool_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `dcop_param_tool_bt`
--

LOCK TABLES `dcop_param_tool_bt` WRITE;
/*!40000 ALTER TABLE `dcop_param_tool_bt` DISABLE KEYS */;
INSERT INTO `dcop_param_tool_bt` VALUES ('AF15T01','10','MAJOR','*','*',1,'DM242','n16','今日累計數',1,1,'DM240','n15','機台總累計數',1,1,'DM710','n121','生產速度',1,1,'R112','n176','機台開關',1,0,NULL,NULL,NULL,1,1,'DM690','n120','噸數',1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,NULL,NULL,NULL,NULL,'2020-02-16 01:19:44.574653'),('AF15Y09','10','MAJOR','*','*',1,'DM514','n07','今日累計數',1,1,'DM512','n06','機台總累計數',1,1,'DM565','n19','生產速度',1,1,'R003','n135','機台開關',1,0,NULL,NULL,NULL,1,1,'DM100','n04','噸數',1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,NULL,NULL,NULL,NULL,'2020-02-16 01:19:44.574653'),('AF35Y01','10','MAJOR','*','*',1,'DM102','n05','今日累計數',1,1,'DM098','n03','機台總累計數',1,1,'DM014','n02','生產速度(?)',1,1,'R003','n165','機台開關',1,0,NULL,NULL,NULL,1,1,'DM830','n45','噸數',1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,0,NULL,NULL,NULL,1,NULL,NULL,NULL,NULL,'2020-02-16 01:19:44.574653');
/*!40000 ALTER TABLE `dcop_param_tool_bt` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:17:07
