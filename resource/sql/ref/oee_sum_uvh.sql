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
-- Table structure for table `oee_sum_uvh`
--

DROP TABLE IF EXISTS `oee_sum_uvh`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `oee_sum_uvh` (
  `report_time` tinyint(4) NOT NULL,
  `cate` tinyint(4) NOT NULL,
  `tool_id` varchar(12) DEFAULT NULL COMMENT 'length extend to 12 for sintering "before/after" omi -- 2020/03/18 lkchena',
  `device` tinyint(4) NOT NULL,
  `avl` tinyint(4) NOT NULL,
  `eff` tinyint(4) NOT NULL,
  `lost` tinyint(4) NOT NULL,
  `down` tinyint(4) NOT NULL,
  `pm` tinyint(4) NOT NULL,
  `hold` tinyint(4) NOT NULL,
  `setup` tinyint(4) NOT NULL,
  `test` tinyint(4) NOT NULL,
  `mon` tinyint(4) NOT NULL,
  `wait` tinyint(4) NOT NULL,
  `off` tinyint(4) NOT NULL,
  `move_act` tinyint(4) NOT NULL,
  `move_prod` tinyint(4) NOT NULL,
  `move_test` tinyint(4) NOT NULL,
  `move_eng` tinyint(4) NOT NULL,
  `move_avg` tinyint(4) NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `oee_sum_uvh`
--

LOCK TABLES `oee_sum_uvh` WRITE;
/*!40000 ALTER TABLE `oee_sum_uvh` DISABLE KEYS */;
/*!40000 ALTER TABLE `oee_sum_uvh` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:27
